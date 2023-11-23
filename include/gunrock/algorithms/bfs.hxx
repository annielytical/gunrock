/**
 * @file bfs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Breadth-First Search algorithm.
 * @date 2020-11-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace bfs {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors;  /// @todo: implement this.
  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;  /// @todo not used.
  thrust::device_vector<int> keep_going;
    
  int n_vertices = this->get_graph().get_number_of_vertices();
  int n_edges = this->get_graph().get_number_of_edges();
  bool switched = false;

  thrust::device_vector<int> new_distances;

  void init() override {
    keep_going.resize(1);
    visited.resize(1);
    new_distances.resize(n_vertices);
    thrust::fill(thrust::device, keep_going.begin(), keep_going.end(), 1);
    thrust::fill(thrust::device, visited.begin(), visited.end(), 0);
  }

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto n_edges = this->get_graph().get_number_of_edges();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(thrust::device, d_distances + this->param.single_source,
                 d_distances + this->param.single_source + 1, 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;
  using csr_t = typename graph::graph_csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  using csc_t = typename graph::graph_csc_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto keep_going = P->keep_going.data().get();
    auto new_distances = P->new_distances.data().get();


    auto search = [distances, single_source, iteration] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {
      auto old_distance =
          math::atomic::min(&distances[neighbor], iteration + 1);
      return (iteration + 1 < old_distance);
    };

    auto backward = [G, distances, iteration, keep_going, new_distances] __device__(vertex_t const& v) -> void {
      
      if (distances[v] != std::numeric_limits<vertex_t>::max()) {
        return;
      }

      edge_t start_edge = G.template get_starting_edge<csc_t>(v);
      edge_t num_neighbors = G.template get_number_of_neighbors<csc_t>(v);

      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = G.template get_source_vertex<csc_t>(e);

        if (distances[u] == iteration) {
          
          new_distances[v] = distances[u] + 1;
          keep_going[0] = 1;
          return;
        }
      }
    };
    
    
    if (!(P->switched) &&  
       (this->active_frontier->get_number_of_elements() < ((P->n_edges) / 14))) {
      thrust::fill(thrust::device, keep_going, keep_going + 1, 1);
      // Execute advance operator on the provided lambda
      operators::advance::execute<operators::load_balance_t::block_mapped>(
          G, E, search, context);
    } else {
      P->switched = true;
      thrust::fill(thrust::device, keep_going, keep_going + 1, 0);
      thrust::copy_n(thrust::device, distances, P->n_vertices, new_distances);
      // Execute advance operator on the provided lambda
      operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
          G,         // graph
          backward,  // lambda function
          context);  // context
    
      thrust::copy_n(thrust::device, new_distances, P->n_vertices, distances);

      
    }
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto f = this->get_enactor()->active_frontier;
    P->visited[0] = P->visited[0] + f->get_number_of_elements(); 
    
    return (P->keep_going[0] == 0) || f->is_empty();
  }

};  // struct enactor_t

/**
 * @brief Run Breadth-First Search algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(single_source);
  result_type result(distances, predecessors);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace bfs
}  // namespace gunrock