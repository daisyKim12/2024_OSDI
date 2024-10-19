#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void search_kernel(int *graph, int *queries, int *results, int topk)
{

  const auto query_id = blockIdx.y;
  
  extern __shared__ std::uint32_t smem[];


  std::uint32_t result_buffer_size    = internal_topk + (search_width * graph_degree);
  std::uint32_t result_buffer_size_32 = result_buffer_size;
  if (result_buffer_size % 32) { result_buffer_size_32 += 32 - (result_buffer_size % 32); }
  const auto small_hash_size = hashmap::get_size(small_hash_bitlen);

  const auto query_smem_buffer_length =
    raft::ceildiv<uint32_t>(dataset_desc.dim, DATASET_BLOCK_DIM) * DATASET_BLOCK_DIM;
  auto query_buffer          = reinterpret_cast<QUERY_T*>(smem);
  auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + query_smem_buffer_length);
  auto result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto visited_hash_buffer =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto parent_list_buffer = reinterpret_cast<INDEX_T*>(visited_hash_buffer + small_hash_size);
  auto distance_work_buffer_ptr =
    reinterpret_cast<std::uint8_t*>(parent_list_buffer + search_width);
  auto topk_ws        = reinterpret_cast<std::uint32_t*>(distance_work_buffer_ptr +
                                                  DATASET_DESCRIPTOR_T::smem_buffer_size_in_byte);
  auto terminate_flag = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto smem_work_ptr  = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // Set smem working buffer for the distance calculation
  dataset_desc.set_smem_ptr(distance_work_buffer_ptr);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  const DATA_T* const query_ptr = queries_ptr + query_id * dataset_desc.dim;
  dataset_desc.template copy_query<DATASET_BLOCK_DIM>(
    query_ptr, query_buffer, query_smem_buffer_length);

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init hashmap
  INDEX_T* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * query_id);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();

  // compute distance to randomly selecting nodes
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  device::compute_distance_to_random_nodes<TEAM_SIZE, DATASET_BLOCK_DIM>(result_indices_buffer,
                                                                         result_distances_buffer,
                                                                         query_buffer,
                                                                         dataset_desc,
                                                                         result_buffer_size,
                                                                         num_distilation,
                                                                         rand_xor_mask,
                                                                         local_seed_ptr,
                                                                         num_seeds,
                                                                         local_visited_hashmap_ptr,
                                                                         hash_bitlen,
                                                                         metric);
  __syncthreads();

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {

      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {

        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (multi_warps_1 || multi_warps_2) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (multi_warps_1 || multi_warps_2) {
            hash_start_tid = 64;
          } else {
            hash_start_tid = 32;
          }
        }
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen, hash_start_tid);
      }

      // topk with bitonic sort
      if (std::is_same<SAMPLE_FILTER_T,
                       raft::neighbors::filtering::none_cagra_sample_filter>::value ||
          *filter_flag == 0) {
        topk_by_bitonic_sort<MAX_ITOPK, MAX_CANDIDATES>(result_distances_buffer,
                                                        result_indices_buffer,
                                                        internal_topk,
                                                        result_distances_buffer + internal_topk,
                                                        result_indices_buffer + internal_topk,
                                                        search_width * graph_degree,
                                                        topk_ws,
                                                        (iter == 0),
                                                        multi_warps_1,
                                                        multi_warps_2);
        __syncthreads();
      } else {
        topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(
          result_distances_buffer,
          result_indices_buffer,
          internal_topk + search_width * graph_degree,
          internal_topk,
          false);
        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
    } else {
      // topk with radix block sort
      topk_by_radix_sort<MAX_ITOPK, INDEX_T>{}(
        internal_topk,
        gridDim.x,
        result_buffer_size,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        nullptr,
        topk_ws,
        true,
        reinterpret_cast<std::uint32_t*>(smem_work_ptr));

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen);
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag,
                                                         parent_list_buffer,
                                                         result_indices_buffer,
                                                         internal_topk,
                                                         dataset_desc.size,
                                                         search_width);
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      hashmap_restore(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk, first_tid);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

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
      hash_bitlen,
      parent_list_buffer,
      result_indices_buffer,
      search_width,
      metric);
    __syncthreads();

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                raft::neighbors::filtering::none_cagra_sample_filter>::value) {
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
      const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }
    iter++;
  }

  // Post process for filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              raft::neighbors::filtering::none_cagra_sample_filter>::value) {
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
    const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

    for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree;
         i += blockDim.x) {
      const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
      if (node_id != (invalid_index & ~index_msb_1_mask) && !sample_filter(query_id, node_id)) {
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        result_indices_buffer[i]   = invalid_index;
      }
    }

    __syncthreads();
    topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(
      result_distances_buffer,
      result_indices_buffer,
      internal_topk + search_width * graph_degree,
      top_k,
      false);
    __syncthreads();
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


torch::Tensor search(torch::Tensor graph, torch::Tensor queries) {
  
  int block_size = 64;
  int num_queries = queries.size(0);
  int topk = 10;

  torch::Tensor results = torch::zeros({num_queries, topk}, torch::device(queries.device()).dtype(torch::kInt32));
  
  dim3 thread_dims(block_size, 1, 1);
  dim3 block_dims(1, num_queries, 1);

  search_kernel<<<block_dims, thread_dims>>>(
      graph.data_ptr<int>(),
      queries.data_ptr<int>(),
      results.data_ptr<int>(),
      topk
  );  

  return results;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search", &search, "A function that performs search on the graph");
}