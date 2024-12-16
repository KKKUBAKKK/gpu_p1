#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "../include/kmeans_gpu1.cuh"
#include <iostream>

#define BLOCK_SIZE 256

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

// Cleanup function
void cleanup_gpu_resources(GPUResources& res) {
    if (res.d_points) cudaFree(res.d_points);
    if (res.d_centroids) cudaFree(res.d_centroids);
    if (res.d_new_centroids) cudaFree(res.d_new_centroids);
    if (res.d_assignments) cudaFree(res.d_assignments);
    if (res.d_assignments_counter) cudaFree(res.d_assignments_counter);
    if (res.d_cluster_sizes) cudaFree(res.d_cluster_sizes);
    if (res.d_cluster_sums) cudaFree(res.d_cluster_sums);
    if (res.d_changed) cudaFree(res.d_changed);
    res = GPUResources(); // Reset to nullptr
}

// CUDA kernel for calculating distances and finding nearest centroids
__global__ void findNearestCentroids(
    const float* points,      // Pointer to the array of points
    const float* centroids,   // Pointer to the array of centroids
    int* assignments,         // Pointer to the array of assignments
    const int N,              // Number of points
    const int K,              // Number of clusters
    const int D,              // Number of dimensions
    int* changed              // Number of changed assignments
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx = threadIdx.x;
    int points_in_block = blockDim.x;

    // Put centroids and points in shared memory
    extern __shared__ float shmem[];

    float* shared_centroids = shmem;
    float* shared_points = &shmem[K * D];

    // Threads in every block with idx.x < K will each put one cluster into shared memory
    if (threadIdx.x < K) {
        for (int i = 0; i < D; i++) {
            shared_centroids[i * K + blockIdx] = centroids[i * K + blockIdx];
        }
    }

    __syncthreads();

    if (idx < N) {
        // Each thread puts it's own point into the shared memory
        for (int i = 0; i < D; i++) {
            shared_points[i * points_in_block + blockIdx] = points[i * N + idx];
        }

        float min_dist = FLT_MAX;
        int nearest_centroid = 0;

        // Iterate over each centroid
        for (int c = 0; c < K; c++) {
            float dist = 0.0f;

            // Calculate the squared Euclidean distance
            for (int d = 0; d < D; d++) {
                float diff = shared_points[d * points_in_block + blockIdx] - shared_centroids[d * K + c];
                dist += diff * diff;
            }

            // Update the nearest centroid if a closer one is found
            if (dist < min_dist) {
                min_dist = dist;
                nearest_centroid = c;
            }
        }

        // Assign the point to the nearest centroid
        if (assignments[idx] != nearest_centroid)
            atomicAdd(changed, 1);
        assignments[idx] = nearest_centroid;
    }
}

// CUDA kernel to sum up all the points for each cluster and dimension
__global__ void sum(
    const float* points,
    const int* assignments,
    int* assignments_counter,
    float* new_centroids,
    const int N,
    const int k,
    const int d
) {
    extern __shared__ float shared_mem[];
    float* shared_sums = shared_mem;
    int* shared_counts = (int*)&shared_sums[k * d];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    for (int i = tid; i < k * d; i += blockDim.x) {
        shared_sums[i] = 0.0f;
    }
    if (tid < k) {
        shared_counts[tid] = 0;
    }
    
    __syncthreads();

    // Process points
    if (gid < N) {
        int cluster = assignments[gid];
        atomicAdd(&shared_counts[cluster], 1);
        
        for (int dim = 0; dim < d; dim++) {
            float point_val = points[dim * N + gid];
            atomicAdd(&shared_sums[dim * k + cluster], point_val);
        }
    }

    __syncthreads();

    // Add shared sums to new centroids in global memory
    for (int i = tid; i < k * d; i += blockDim.x) {
        int cluster = i % k;
        int dim = i / k;
        if (shared_sums[dim * k + cluster] != 0.0f) {
            atomicAdd(&new_centroids[dim * k + cluster], 
                     shared_sums[dim * k + cluster]);
        }
    }

    // Add shared counts to assignments counter in global memory
    if (tid < k && shared_counts[tid] > 0) {
        atomicAdd(&assignments_counter[tid], shared_counts[tid]);
    }
}

// CUDA kernel to assign the new centroids divided by assignments counters to the centroids
__global__ void update(
    float* centroids,
    float* new_centroids,
    const int* assignments_counter,
    const int K,
    const int D
) {
    const int tid = threadIdx.x;
    const int total_elements = K * D;
    
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int cluster = idx % K;
        int dim = idx / K;
        
        if (assignments_counter[cluster] > 0) {
            centroids[dim * K + cluster] = 
                new_centroids[dim * K + cluster] / assignments_counter[cluster];
        }
    }
}

// First version of gpu K-Means algorithm
void kmeans_gpu1(
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

    // Allocate device memory
    GPUResources res;
    
    CUDA_CHECK(cudaMalloc(&res.d_points, N * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_centroids, k * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_assignments, N * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_changed, sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_assignments_counter, k * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_new_centroids, k * d * sizeof(float)), res);
    
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

    // Configure kernel launch parameters
    dim3 block_size(BLOCK_SIZE);
    dim3 num_blocks_points((N + block_size.x - 1) / block_size.x);
    dim3 num_blocks_centroids((k * d + block_size.x - 1) / block_size.x);

    // Calculate shared memory size
    size_t shared_mem_size = (k * d + BLOCK_SIZE * d) * sizeof(float);
    size_t shared_mem_size_sum = (k * d) * sizeof(float) + k * sizeof(int);

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

        // Find nearest centroids
        cudaEventRecord(start_kernel1);
        findNearestCentroids<<<num_blocks_points, block_size, shared_mem_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, N, k, d, res.d_changed);
        
        CUDA_CHECK(cudaGetLastError(), res);
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        cudaEventRecord(stop_kernel1);
        cudaEventSynchronize(stop_kernel1);
        cudaEventElapsedTime(&milliseconds, start_kernel1, stop_kernel1);
        std::cout << "Assigning nearest centroids execution time: " << milliseconds << " ms" << std::endl;

        // Copy the number of changed assignments back to host
        CUDA_CHECK(cudaMemcpy(&changed, res.d_changed, sizeof(int), cudaMemcpyDeviceToHost), res);

        // Check if any assignments changed
        if (changed == 0) {
            std::cout << "No changes in assignments, stopping the algorithm" << std::endl;
            break;
        }

        // Reset assignemnts counters and new centroids
        CUDA_CHECK(cudaMemset(res.d_assignments_counter, 0, k * sizeof(int)), res);
        CUDA_CHECK(cudaMemset(res.d_new_centroids, 0, k * d * sizeof(float)), res);
        
        // Update centroids
        cudaEventRecord(start_kernel2);

        // Sum all points for all clusters and dimensions using sum kernel
        sum<<<num_blocks_points, block_size, shared_mem_size_sum>>>(
            res.d_points, res.d_assignments, res.d_assignments_counter, res.d_new_centroids, N, k, d);

        CUDA_CHECK(cudaGetLastError(), res);
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        // Update centroids by dividing the sums by the number of points in each cluster
        update<<<1, block_size>>>(res.d_centroids, res.d_new_centroids, res.d_assignments_counter, k, d);

        CUDA_CHECK(cudaGetLastError(), res);
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        cudaEventRecord(stop_kernel2);

        cudaEventSynchronize(stop_kernel2);
        cudaEventElapsedTime(&milliseconds, start_kernel2, stop_kernel2);
        std::cout << "Updating centroids execution time: " << milliseconds << " ms" << std::endl;
    
        // Display general info about iteration
        cudaEventRecord(stop_it);
        cudaEventSynchronize(stop_it);
        cudaEventElapsedTime(&milliseconds, start_it, stop_it);
        std::cout << "Iteration " << iter << " completed in " << milliseconds << " ms" << std::endl;
        std::cout << "Points that changed cluster: " << changed << std::endl;

        // Setting changed back to 0
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
    cleanup_gpu_resources(res);
}