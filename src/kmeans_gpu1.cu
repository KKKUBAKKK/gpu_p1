#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "kmeans_gpu1.h"

// TODO: use shared memory to optimize the kernels
// TODO: use accumulation to sum up the points assigned to each cluster (be smart about it)
// TODO: think about threads per block and blocks per grid
// TODO: think about threads assignment
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
    const int n_dims          // Number of dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_points) {
        float min_dist = FLT_MAX;
        int nearest_centroid = 0;

        // Iterate over each centroid
        for (int c = 0; c < n_clusters; c++) {
            float dist = 0.0f;

            // Calculate the squared Euclidean distance
            for (int d = 0; d < n_dims; d++) {
                float diff = points[d * n_points + idx] - centroids[d * n_clusters + c];
                dist += diff * diff;
            }

            // Update the nearest centroid if a closer one is found
            if (dist < min_dist) {
                min_dist = dist;
                nearest_centroid = c;
            }
        }

        // Assign the point to the nearest centroid
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

        float sum = 0.0f;
        int count = 0;

        // Sum up all points assigned to the cluster 'c' for dimension 'd'
        for (int i = 0; i < n_points; i++) {
            if (assignments[i] == c) {
                sum += points[d * n_points + i];
                count++;
            }
        }

        // Update cluster size
        cluster_sizes[c] = count;

        // Calculate the new centroid position if the cluster has points assigned
        if (count > 0) {
            centroids[idx] = sum / count;
        }
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
    // Allocate device memory
    GPUResources res;
    
    CUDA_CHECK(cudaMalloc(&res.d_points, N * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_centroids, k * d * sizeof(float)), res);
    CUDA_CHECK(cudaMalloc(&res.d_assignments, N * sizeof(int)), res);
    CUDA_CHECK(cudaMalloc(&res.d_cluster_sizes, k * sizeof(int)), res);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(res.d_points, h_points, N * d * sizeof(float), cudaMemcpyHostToDevice), res);
    CUDA_CHECK(cudaMemcpy(res.d_centroids, h_centroids, k * d * sizeof(float), cudaMemcpyHostToDevice), res);
    
    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 num_blocks_points((N + block_size.x - 1) / block_size.x);
    dim3 num_blocks_centroids((k * d + block_size.x - 1) / block_size.x);
    
    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Find nearest centroids
        findNearestCentroids<<<num_blocks_points, block_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, N, k, d);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), res);
        
        // Wait for kernel to finish and check for errors
        CUDA_CHECK(cudaDeviceSynchronize(), res);
        
        // Update centroids
        updateCentroids<<<num_blocks_centroids, block_size>>>(
            res.d_points, res.d_centroids, res.d_assignments, res.d_cluster_sizes,
            N, k, d);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), res);
        
        // Wait for kernel to finish and check for errors
        CUDA_CHECK(cudaDeviceSynchronize(), res);
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_centroids, res.d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost), res);
    CUDA_CHECK(cudaMemcpy(h_assignments, res.d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), res);
    
    // Cleanup
    CUDA_CHECK(cudaFree(res.d_points), res);
    CUDA_CHECK(cudaFree(res.d_centroids), res);
    CUDA_CHECK(cudaFree(res.d_assignments), res);
    CUDA_CHECK(cudaFree(res.d_cluster_sizes), res);
}