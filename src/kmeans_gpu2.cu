#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "../include/kmeans_gpu1.cuh"
#include <iostream>

#define BLOCK_SIZE 256

// TODO: add timing information
// TODO: fix the makefile

// Macro for checking CUDA errors
#define CUDA_CHECK(call, res) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            cleanup_gpu_resources(res); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// // Cleanup function
// static void cleanup_gpu_resources(GPUResources& res) {
//     if (res.d_points) cudaFree(res.d_points);
//     if (res.d_centroids) cudaFree(res.d_centroids);
//     if (res.d_assignments) cudaFree(res.d_assignments);
//     if (res.d_cluster_sizes) cudaFree(res.d_cluster_sizes);
//     res = GPUResources(); // Reset to nullptr
// }

// Kernel to find nearest centroids and update assignments
__global__ void kmeans_iteration(
    const float* points,        // Array of points [N x d]
    float* centroids,           // Array of centroids [k x d]
    int* assignments,           // Array of assignments [N]
    int* cluster_sizes,         // Array of cluster sizes [num_blocks x k]
    float* cluster_sums,        // Array of cluster sums [num_blocks x k x d]
    const int N,                // Number of points
    const int d,                // Number of dimensions
    const int k,                // Number of clusters
    int* changed                // Flag to indicate if any assignments changed
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sidx = threadIdx.x;
    int points_in_block = blockDim.x;

    // Put centroids and points in shared memory
    extern __shared__ float shmem[];

    float* shared_centroids = shmem;
    float* shared_sums = &shmem[k * d];
    float* shared_points = &shared_sums[k * d];
    int* shared_counts = (int*) &shared_points[points_in_block * d];

    // Threads in every block with idx.x < n_clusters will each put one cluster into shared memory
    if (threadIdx.x < k) {
        for (int i = 0; i < d; i++) {
            shared_centroids[i * k + threadIdx.x] = centroids[i * k + threadIdx.x];
            shared_sums[i * k + threadIdx.x] = 0.0f;
        }
        shared_counts[threadIdx.x] = 0;
    }

    __syncthreads();

    if (idx < N) {
        // Each thread puts it's own point into the shared memory
        for (int i = 0; i < d; i++) {
            shared_points[i * points_in_block + sidx] = points[i * N + idx];
        }

        float min_dist = FLT_MAX;
        int nearest_centroid = 0;

        // Iterate over each centroid
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            
            // Calculate the squared Euclidean distance
            for (int dim = 0; dim < d; dim++) {
                float diff = shared_points[dim * points_in_block + sidx] - shared_centroids[dim * k + c];
                dist += diff * diff;
            }

            // Update the nearest centroid if a closer one is found
            if (dist < min_dist) {
                min_dist = dist;
                nearest_centroid = c;
            }
        }

        // Assign the point to the nearest centroid
        if (assignments[idx] != nearest_centroid) {
            atomicAdd(changed, 1);
            assignments[idx] = nearest_centroid;
        }
        // printf("Point %d assigned to cluster %d\n", idx, nearest_centroid);
        atomicAdd(&shared_counts[nearest_centroid], 1);

        // Update the sum
        for (int i = 0; i < d; i++) {
            atomicAdd(&shared_sums[i * k + nearest_centroid], shared_points[i * points_in_block + sidx]);
        }
    }

    __syncthreads();

    // TODO: think if the indexing is correct
    // Store the results back to global memory
    if (threadIdx.x < k) {
        int block_offset = blockIdx.x * k;
        atomicAdd(&cluster_sizes[block_offset + threadIdx.x], shared_counts[threadIdx.x]);
        for (int i = 0; i < d; i++) {
            atomicAdd(&cluster_sums[i * k * blockIdx.x + block_offset + threadIdx.x], shared_sums[i * k + threadIdx.x]);
        }
    }
}

// TODO: use shared memory or parallelize more
// Kernel to update centroids
__global__ void computeFinalCentroids(
    float* centroids,           // [k * d]
    const float* cluster_sums,  // [num_blocks * k * d]
    const int* cluster_sizes,   // [num_blocks * k]
    const int num_blocks,
    const int k,
    const int d
) {
    // Each thread handles one dimension of one centroid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * d) return;

    int centroid_idx = idx / d;    // Which centroid
    int dim_idx = idx % d;         // Which dimension

    // Sum up partial results
    float sum = 0.0f;
    int count = 0;

    for (int block = 0; block < num_blocks; block++) {
        // Get partial sum for this centroid/dimension from this block
        sum += cluster_sums[num_blocks * k * dim_idx + block * k + centroid_idx];
        
        // Get partial count for this centroid from this block
        count += cluster_sizes[block * k + centroid_idx];
    }

    // Update centroid if count > 0
    if (count > 0) {
        centroids[dim_idx * k + centroid_idx] = sum / count;
    }
}

void kmeans_gpu2(
    const float* h_points,    // Pointer to the array of points on the host
    float* h_centroids,       // Pointer to the array of centroids on the host
    int* h_assignments,       // Pointer to the array of assignments on the host
    const int N,              // Number of points
    const int d,              // Number of dimensions
    const int k,              // Number of clusters
    const int max_iter        // Maximum number of iterations
)
{
    // Create CUDA events for timing
    cudaEvent_t start_it, stop_it, start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_it);
    cudaEventCreate(&stop_it);
    float milliseconds = 0.0f;
    int changed = 0;

    // Configure kernel launch parameters
    dim3 block_size(BLOCK_SIZE);
    dim3 num_blocks_points((N + block_size.x - 1) / block_size.x);
    dim3 num_blocks_centroids((k * d + block_size.x - 1) / block_size.x);

    // Calculate shared memory size
    size_t shared_mem_size = (k * d + k * d + BLOCK_SIZE * d) * sizeof(float) + k * sizeof(int);

    // Allocate device memory
    GPUResources res;
    
    CUDA_CHECK(cudaMalloc(&res.d_points, N * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_centroids, k * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_assignments, N * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sizes, k * num_blocks_points.x * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sums, k * d * num_blocks_points.x * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_changed, sizeof(int)), res);
    
    // Copy data to device
    cudaEvent_t start_copy, stop_copy;
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);
    cudaEventRecord(start_copy);

    CUDA_CHECK(cudaMemcpy(res.d_points, h_points, N * d * sizeof(float), cudaMemcpyHostToDevice), res);
    CUDA_CHECK(cudaMemcpy(res.d_centroids, h_centroids, k * d * sizeof(float), cudaMemcpyHostToDevice), res);
    CUDA_CHECK(cudaMemcpy(res.d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice), res);
    
    cudaEventRecord(stop_copy);
    cudaEventSynchronize(stop_copy);
    cudaEventElapsedTime(&milliseconds, start_copy, stop_copy);
    std::cout << "Data copying to device: " << milliseconds << " ms" << std::endl;

    // Create events for timing
    cudaEvent_t start_kernel1, stop_kernel1, start_kernel2, stop_kernel2;
    cudaEventCreate(&start_kernel1);
    cudaEventCreate(&stop_kernel1);
    cudaEventCreate(&start_kernel2);
    cudaEventCreate(&stop_kernel2);
    
    // Main loop
    cudaEventRecord(start);
    for (int iter = 0; iter < max_iter; iter++) {
        cudaEventRecord(start_it);

        // Find nearest centroids and get sums for each block
        cudaEventRecord(start_kernel1);
        kmeans_iteration<<<num_blocks_points, block_size, shared_mem_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, res.d_cluster_sizes,
            res.d_cluster_sums, N, d, k, res.d_changed);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), res);
        
        // Wait for kernel to finish and check for errors
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        cudaEventRecord(stop_kernel1);

        // Display kernel timing information
        cudaEventSynchronize(stop_kernel1);
        cudaEventElapsedTime(&milliseconds, start_kernel1, stop_kernel1);
        std::cout << "Assigning nearest centroids execution time: " << milliseconds << " ms" << std::endl;

        // Copy the flag to the host
        CUDA_CHECK(cudaMemcpy(&changed, res.d_changed, sizeof(int), cudaMemcpyDeviceToHost), res);

        // Check if any assignments changed
        if (changed == 0) {
            std::cout << "No changes in assignments, stopping the algorithm" << std::endl;
            break;
        }
        
        // Update centroids
        cudaEventRecord(start_kernel2);
        computeFinalCentroids<<<num_blocks_centroids, block_size>>>(
            res.d_centroids, res.d_cluster_sums, res.d_cluster_sizes, num_blocks_points.x, k, d);
        cudaEventRecord(stop_kernel2);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), res);
        
        // Wait for kernel to finish and check for errors
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        // Display kernel timing information
        cudaEventSynchronize(stop_kernel2);
        cudaEventElapsedTime(&milliseconds, start_kernel2, stop_kernel2);
        std::cout << "Updating centroids execution time: " << milliseconds << " ms" << std::endl;
    
        // Display general info about iteration
        cudaEventRecord(stop_it);
        cudaEventSynchronize(stop_it);
        cudaEventElapsedTime(&milliseconds, start_it, stop_it);
        std::cout << "Iteration " << iter << " completed in " << milliseconds << " ms" << std::endl;
        std::cout << "Points that changed cluster: " << changed << std::endl;

        // Reset the flag
        changed = 0;
        CUDA_CHECK(cudaMemcpy(res.d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice), res);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total execution time of the main loop: " << milliseconds << " ms" << std::endl;
    
    // Copy results back to host
    cudaEvent_t start_copy_back, stop_copy_back;
    cudaEventCreate(&start_copy_back);
    cudaEventCreate(&stop_copy_back);
    cudaEventRecord(start_copy_back);

    CUDA_CHECK(cudaMemcpy(h_centroids, res.d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost), res);
    CUDA_CHECK(cudaMemcpy(h_assignments, res.d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), res);

    cudaEventRecord(stop_copy_back);
    cudaEventSynchronize(stop_copy_back);
    cudaEventElapsedTime(&milliseconds, start_copy_back, stop_copy_back);
    std::cout << "Data copying back to host: " << milliseconds << " ms" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(res.d_points), res);
    CUDA_CHECK(cudaFree(res.d_centroids), res);
    CUDA_CHECK(cudaFree(res.d_assignments), res);
    CUDA_CHECK(cudaFree(res.d_cluster_sizes), res);
    CUDA_CHECK(cudaFree(res.d_cluster_sums), res);
    CUDA_CHECK(cudaFree(res.d_changed), res);
}