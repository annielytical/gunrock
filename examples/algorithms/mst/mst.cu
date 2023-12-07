#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/mst.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/performance.hxx>
#include "mst_cpu.hxx"  // Reference implementation
#include <cxxopts.hpp>
#include <iomanip>

using namespace gunrock;
using namespace memory;

void test_mst(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Minimum Spanning Tree");

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  if (!properties.symmetric) {
    printf("Error: input matrix must be symmetric\n");
    exit(1);
  }

  csr.from_coo(coo);

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> mst_weight(1);

  // --
  // GPU Run

  std::vector<float> run_times;
  auto benchmark_metrics =
      std::vector<benchmark::host_benchmark_t>(params.num_runs);
  
  for (int i = 0; i < params.num_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::mst::run(G, mst_weight.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    // Placeholder since MST does not use sources
    std::vector<int> src_placeholder;

    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "mst",
        params.filename, "market", params.json_dir, params.json_file,
        src_placeholder, tag_vect, num_arguments, argument_array);
  }

  // Print info for last run
  thrust::host_vector<weight_t> h_mst_weight = mst_weight;
  std::cout << "GPU MST Weight: " << std::fixed << std::setprecision(4)
            << h_mst_weight[0] << std::endl;
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    weight_t cpu_mst_weight;
    float cpu_elapsed =
        mst_cpu::run<csr_t, vertex_t, edge_t, weight_t>(csr, &cpu_mst_weight);
    std::cout << "CPU MST Weight: " << std::fixed << std::setprecision(4)
              << cpu_mst_weight << std::endl;
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  }
}

int main(int argc, char** argv) {
  test_mst(argc, argv);
}