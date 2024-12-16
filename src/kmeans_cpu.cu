#include "../include/kmeans_cpu.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <new>
#include <iostream>

void kmeans_cpu(
    const float* points,          // Input points array [N x D]
    float* centroids,             // Centroids array [k x D]
    int* assigned_clusters,       // Cluster assignments [N]
    const int N,                  // Number of points
    const int D,                  // Dimensions
    const int k,                  // Number of clusters
    const int max_iter            // Maximum iterations
    ) 
{
    // Allocate temporary arrays
    int* cluster_sizes = nullptr;
    try {
        cluster_sizes = new int[k];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for cluster_sizes: " << e.what() << std::endl;
        exit(1);
    }

    float* new_centroids = nullptr;
    try {
        new_centroids = new float[k * D];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for new_centroids: " << e.what() << std::endl;
        exit(1);
    }

    int iter = 0;
    int changed = 1;
    while (iter < max_iter && changed > 0) {
        changed = 0;

        // Start timer for this iteration
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; i++) {
            float min_distance = INFINITY;
            int n = 0;
            for (int j = 0; j < k; j++) {
                float dist = calculate_distance(i, j, points, centroids, N, k, D);
                if (min_distance > dist) {
                    min_distance = dist;
                    n = j;
                }
            }
            if (assigned_clusters[i] != n) {
                changed++;
                assigned_clusters[i] = n;
            }
            for (int d = 0; d < D; d++) {
                new_centroids[d * k + n] += points[d * N + i];
            }
            cluster_sizes[n]++;
        }
        for (int j = 0; j < k; j++) {
            if (cluster_sizes[j] > 0) {
                for (int d = 0; d < D; d++) {
                    centroids[d * k + j] = new_centroids[d * k + j] / cluster_sizes[j];
                    new_centroids[d * k + j] = 0;
                }
                cluster_sizes[j] = 0;
            }
        }

        // End timer and print results
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Iteration " << iter << " completed, " << changed << " points changed in time: " << duration.count() << " ms" << std::endl;

        iter++;
    }
}

float calculate_distance(
    const int& point,        // Input point coordinates
    const int& centroid,     // Centroid coordinates
    const float* points,     // Points array
    const float* centroids,  // Centroids array
    const int N,             // Number of points
    const int k,             // Number of centroids
    const int D              // Number of dimensions
    )
{
    float dist = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = points[i * N + point] - centroids[i * k + centroid];
        dist += diff * diff;
    }
    return dist;
}