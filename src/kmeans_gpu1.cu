#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "../include/kmeans_gpu1.cuh"
#include <iostream>

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

// Cleanup function
static void cleanup_gpu_resources(GPUResources& res) {
    if (res.d_points) cudaFree(res.d_points);
    if (res.d_centroids) cudaFree(res.d_centroids);
    if (res.d_assignments) cudaFree(res.d_assignments);
    if (res.d_cluster_sizes) cudaFree(res.d_cluster_sizes);
    res = GPUResources(); // Reset to nullptr
}

// CUDA kernel for calculating distances and finding nearest centroids
__global__ void findNearestCentroids(
    const float* points,      // Pointer to the array of points
    const float* centroids,   // Pointer to the array of centroids
    int* assignments,         // Pointer to the array of assignments
    const int n_points,       // Number of points
    const int n_clusters,     // Number of clusters
    const int n_dims,         // Number of dimensions
    int* changed              // Number of changed assignments
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sidx = threadIdx.x;
    int points_in_block = blockDim.x;

    // Put centroids and points in shared memory
    __shared__ float shared_centroids[n_clusters * n_dims];
    __shared__ float shared_points[points_in_block * n_dims];

    // Threads in every block with idx.x < n_clusters will each put one cluster into shared memory
    if (threadIdx.x < n_clusters) {
        for (int i = 0; i < n_dims; i++) {
            shared_centroids[i * n_clusters + threadIdx.x] = centroids[i * n_clusters + threadIdx.x];
        }
    }

    __syncthreads();

    if (idx < n_points) {
        // Each thread puts it's own point into the shared memory
        for (int i = 0; i < n_dims; i++) {
            shared_points[i * points_in_block + sidx] = points[i * n_points + idx];
        }

        float min_dist = FLT_MAX;
        int nearest_centroid = 0;

        // Iterate over each centroid
        for (int c = 0; c < n_clusters; c++) {
            float dist = 0.0f;

            // Calculate the squared Euclidean distance
            for (int d = 0; d < n_dims; d++) {
                float diff = shared_points[d * points_in_block + sidx] - shared_centroids[d * n_clusters + c];
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

// CUDA kernel for updating centroids
__global__ void updateCentroids(
    const float* points,       // Pointer to the array of points
    float* centroids,          // Pointer to the array of centroids
    const int* assignments,    // Pointer to the array of assignments
    int* cluster_sizes,        // Pointer to the array of cluster sizes
    const int n_points,        // Number of points
    const int n_clusters,      // Number of clusters
    const int n_dims           // Number of dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_clusters * n_dims) {
        int c = idx / n_dims;  // Cluster index
        int d = idx % n_dims;  // Dimension index

        // Sum up all points assigned to the cluster 'c' for dimension 'd' using sumPoints kernel
        dim3 block_size(256);
        dim3 num_blocks_points((n_points + block_size.x - 1) / block_size.x);
        sumPoints<<<num_blocks_points, block_size>>>(c, d, points, assignments, centroids, n_points, n_clusters, n_dims);
    }
}

// CUDA kernel to sum up one dimension of the points for one cluster
__global__ void sumPoints(
    const int cluster,         // Index of the cluster
    const int dimension,       // Index of the dimension
    const float* points,       // Pointer to the array of points
    const int* assignments,    // Pointer to the array of assignments
    float* centroids,          // Pointer to the array of centroids
    const int n_points,        // Number of points
    const int n_clusters,      // Number of clusters
    const int n_dims           // Number of dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Move points to shared memory (set to 0 if not assigned to the cluster)
    __shared__ float shared_points[blockDim.x];
    __shared__ int assignments_counter = 0;
    if (idx < n_points) {
        if (assignments[idx] == cluster) {
            atomicAdd(&assignments_counter, 1);
            shared_points[threadIdx.x] = points[dimension * n_points + idx];
        }
        else {
            shared_points[threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    if (assignments_counter == 0) {
        return;
    }

    // Sum up one dimension of all the points from shared memory in parallel
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            shared_points[threadIdx.x] += shared_points[threadIdx.x + i];
            shared_points[threadIdx.x + i] = 0.0f;
        }
        if (threadIdx.x == 0 && i + i < blockDim.x) {
            shared_points[0] += shared_points[i + i];
            shared_points[i + i] = 0.0f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        centroids[dimension * n_clusters + cluster] = shared_points[0] / assignments_counter;
    }
}

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

    // Allocate device memory
    GPUResources res;
    
    CUDA_CHECK(cudaMalloc(&res.d_points, N * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_centroids, k * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_assignments, N * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sizes, k * sizeof(int)), res);
    
    // Copy data to device
    cudaEvent_t start_copy, stop_copy;
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);
    cudaEventRecord(start_copy);

    CUDA_CHECK(cudaMemcpy(res.d_points, h_points, N * d * sizeof(float), cudaMemcpyHostToDevice), res);
    CUDA_CHECK(cudaMemcpy(res.d_centroids, h_centroids, k * d * sizeof(float), cudaMemcpyHostToDevice), res);
    
    cudaEventRecord(stop_copy);
    cudaEventSynchronize(stop_copy);
    cudaEventElapsedTime(&milliseconds, start_copy, stop_copy);
    std::cout << "Data copying to device: " << milliseconds << " ms" << std::endl;

    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 num_blocks_points((N + block_size.x - 1) / block_size.x);
    dim3 num_blocks_centroids((k * d + block_size.x - 1) / block_size.x);

    // Create events for timing
    cudaEvent_t start_kernel1, stop_kernel1, start_kernel2, stop_kernel2;
    cudaEventCreate(&start_kernel1);
    cudaEventCreate(&stop_kernel1);
    cudaEventCreate(&start_kernel2);
    cudaEventCreate(&stop_kernel2);
    
    // Main loop
    cudaEventRecord(start);
    int changed = 0;
    for (int iter = 0; iter < max_iter; iter++) {
        cudaEventRecord(start_it);

        // Find nearest centroids
        cudaEventRecord(start_kernel1);
        findNearestCentroids<<<num_blocks_points, block_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, N, k, d, &changed);
        cudaEventRecord(stop_kernel1);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), res);
        
        // Wait for kernel to finish and check for errors
        CUDA_CHECK(cudaDeviceSynchronize(), res);

        // Display kernel timing information
        cudaEventSynchronize(stop_kernel1);
        cudaEventElapsedTime(&milliseconds, start_kernel1, stop_kernel1);
        std::cout << "Assigning nearest centroids execution time: " << milliseconds << " ms" << std::endl;

        // Check if any assignments changed
        if (changed == 0) {
            std::cout << "No changes in assignments, stopping the algorithm" << std::endl;
            break;
        }
        
        // Update centroids
        cudaEventRecord(start_kernel2);
        updateCentroids<<<num_blocks_centroids, block_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, res.d_cluster_sizes,
            N, k, d);
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

        changed = 0;
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
}