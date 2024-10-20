#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

namespace py = pybind11;

#define NUM_QUERIES 1000
#define TOPK 10
#define SEARCH_WIDTH 1
#define MAX_ITER 71
#define MIN_ITER 0
#define INTERNEL_TOPK 64
#define VECTOR_DIM 128
#define DATASET_SIZE 1000000
#define TEAM_SIZE 8
#define GRAPH_DEGREE 64
#define HASH_BITLEN 8

typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

__global__ void search_kernel 
(
  INDEX_T* graph_ptr,
  DATA_T* dataset_ptr,
  DATA_T* queries_ptr,
  INDEX_T* results_ptr,
)
{

  int result_buffer_size = INTERNEL_TOPK + (SEARCH_WIDTH * GRAPH_DEGREE);
  if(result_buffer_size % 32) { result_buffer_size += 32 - (result_buffer_size % 32); }
  // use smem as buffer
  extern __shared__ DATA_T query_buffer[VECTOR_DIM];
  extern __shared__ INDEX_T result_indices_buffer[result_buffer_size];
  extern __shared__ DISTANCE_T result_distances_buffer[result_buffer_size];
  extern __shared__ INDEX_T parent_list_buffer[SEARCH_WIDTH];


  // copy query to smem
  int query_id = blockIdx.y;
  DATA_T* query_ptr = queries_ptr + query_id * VECTOR_DIM;
  for(int i = threadIdx.x; i < VECTOR_DIM; i += blockDim.x) {
    query_buffer[i] = query_ptr[i];
  }
  __syncthreads();

  // init termination flag
  uint32_t termination_flag;
  if (threadIdx.x == 0) {
    terminate_flag = 0;
  }

  // init hashmap need hashmap
  //
  //

  // compute distance to randomly selecting nodes
  //
  //
  __syncthreads();

  std::uint32_t iter = 0;
  while (1) {

    // bitonic sort
    __syncthreads();

    if (iter + 1 == MAX_ITER
  ) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag,
                                                         parent_list_buffer,
                                                         result_indices_buffer,
                                                         internal_topk,
                                                         dataset_desc.size,
                                                         search_width);
    }

    if (*terminate_flag && iter >= MIN_ITER) { break; }

    // compute the norms between child nodes and query node
    constexpr unsigned max_n_frags = 8;
    device::compute_distance_to_child_nodes<TEAM_SIZE, DATASET_BLOCK_DIM, max_n_frags>(
      result_indices_buffer + internal_topk,
      result_distances_buffer + internal_topk,
      query_buffer,
      dataset_desc,
      knn_graph,
      graph_degree,
      local_visited_hashmap_ptr,
      HASH_BITLEN,
      parent_list_buffer,
      result_indices_buffer,
      search_width,
      metric);
    __syncthreads();

    // filtering
    //
    //

    iter++;
  }

  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

    result_indices_ptr[j] =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
}


torch::Tensor search(torch::Tensor graph, torch::Tensor dataset, torch::Tensor queries) {
  
  int block_size = 64;
  int NUM_QUERIES = queries.size(0);
  int TOPK = 10;

  torch::Tensor results = torch::zeros({NUM_QUERIES, TOPK}, torch::device(queries.device()).dtype(torch::kInt32));


  dim3 thread_dims(block_size, 1, 1);
  dim3 block_dims(1, NUM_QUERIES, 1);

  search_kernel<<<block_dims, thread_dims>>>(
    graph.data_ptr<u32int_t>(),
    dataset.data_ptr<float>(),
    queries.data_ptr<float>(),
    results.data_ptr<u32int_t>(),
  );  

  return results;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search", &search, "A function that performs search on the graph");
}