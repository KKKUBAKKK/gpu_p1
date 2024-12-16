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

// Kernel to find nearest centroids and update assignments
__global__ void kmeans_iteration(
    const float* points,        // Array of points
    float* centroids,           // Array of centroids
    int* assignments,           // Array of assignments
    int* cluster_sizes,         // Array of cluster sizes
    float* cluster_sums,        // Array of cluster sums
    const int N,                // Number of points
    const int d,                // Number of dimensions
    const int k,                // Number of clusters
    int* changed,               // Flag to indicate if any assignments changed
    int num_blocks              // Number of blocks
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

    __shared__ int shared_changed;

    // Threads in every block with idx.x < n_clusters will each put one cluster into shared memory
    if (threadIdx.x < k) {
        for (int i = 0; i < d; i++) {
            shared_centroids[i * k + threadIdx.x] = centroids[i * k + threadIdx.x];
            shared_sums[i * k + threadIdx.x] = 0.0f;
        }
        shared_counts[threadIdx.x] = 0;
    }

    if (threadIdx.x == 0) {
        shared_changed = 0;
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
            atomicAdd(&shared_changed, 1);
            assignments[idx] = nearest_centroid;
        }
        atomicAdd(&shared_counts[nearest_centroid], 1);

        // Update the sum
        for (int i = 0; i < d; i++) {
            atomicAdd(&shared_sums[i * k + nearest_centroid], shared_points[i * points_in_block + sidx]);
        }
    }

    __syncthreads();

    // Add changed to global memory
    if (threadIdx.x == 0) {
        atomicAdd(changed, shared_changed);
    }

    // Add shared counts to global cluster sizes
    if (threadIdx.x < k) {
        atomicAdd(&cluster_sizes[threadIdx.x], shared_counts[threadIdx.x]);
    }

    // Add shared sums to global cluster sums
    for (int i = threadIdx.x; i < k * d; i += blockDim.x) {
        int centroid = i % k;
        int dim = i / k;

        atomicAdd(&cluster_sums[dim * k + centroid], shared_sums[dim * k + centroid]);
    }
}


// CUDA kernel that assigns the centroids values from new centroids divided by assignements counter
__global__ void updateCentroids(
    float* centroids,
    float* new_centroids,
    const int* assignments_counter,
    const int k,
    const int d
) {
    // Since k and d are small (max 20x20),
    // we can use a single block with enough threads
    const int tid = threadIdx.x;
    const int total_elements = k * d;
    
    // Each thread handles multiple elements if needed
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int cluster = idx % k;
        int dim = idx / k;
        
        if (assignments_counter[cluster] > 0) {
            centroids[dim * k + cluster] = 
                new_centroids[dim * k + cluster] / assignments_counter[cluster];
        }
    }
}

// Second version of gpu KMeans algorithm
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
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sizes, k * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sums, k * d * sizeof(float)), res);
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

        // Reset the values of cluster sizes and sums
        CUDA_CHECK(cudaMemset(res.d_cluster_sizes, 0, k * sizeof(int)), res);
        CUDA_CHECK(cudaMemset(res.d_cluster_sums, 0, k * d * sizeof(float)), res);

        cudaEventRecord(start_it);

        // Find nearest centroids and get sums for each block
        cudaEventRecord(start_kernel1);
        kmeans_iteration<<<num_blocks_points, block_size, shared_mem_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, res.d_cluster_sizes,
            res.d_cluster_sums, N, d, k, res.d_changed, num_blocks_points.x);
        
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
        updateCentroids<<<1, k * d>>>(
            res.d_centroids, res.d_cluster_sums, res.d_cluster_sizes, k, d);
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
    
    cudaEvent_t start_copy_back, stop_copy_back;
    cudaEventCreate(&start_copy_back);
    cudaEventCreate(&stop_copy_back);
    cudaEventRecord(start_copy_back);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_centroids, res.d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost), res);
    CUDA_CHECK(cudaMemcpy(h_assignments, res.d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), res);

    cudaEventRecord(stop_copy_back);
    cudaEventSynchronize(stop_copy_back);
    cudaEventElapsedTime(&milliseconds, start_copy_back, stop_copy_back);
    std::cout << "Data copying back to host: " << milliseconds << " ms" << std::endl;
    
    // Cleanup
    cleanup_gpu_resources(res);
}