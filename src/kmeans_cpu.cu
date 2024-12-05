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
    
    // Initialize variables
    int changed = 1;
    int iter = 0;

    // Main loop
    while (changed > 0 && iter < max_iter) {
        // Reset changed counter
        changed = 0;

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // Reset temporary arrays
        memset(cluster_sizes, 0, k * sizeof(int));
        memset(new_centroids, 0, k * D * sizeof(float));

        // For each point find nearest centroids
        assign_clusters(points, centroids, new_centroids, assigned_clusters, 
                        cluster_sizes, changed, N, D, k);

        // Update centroids
        update_centroids(new_centroids, cluster_sizes, centroids, D, k);

        // Stop timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Iteration " << iter << " completed, " << changed << " points changed in time: " << duration.count() << " ms" << std::endl;
        
        iter++;
    }

    // Clean up
    delete[] cluster_sizes;
    delete[] new_centroids;
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

int find_closest_centroid(
    const int& point,       // Input point coordinates
    const float* points,    // Array of points
    const float* centroids, // Array of centroids
    const int N,            // Number of points
    const int k,            // Number of centroids
    const int D             // Number of dimensions
    )
{
    float min_distance = INFINITY;
    int best_cluster = 0;

    for (int j = 0; j < k; j++) {
        float distance = calculate_distance(point, j, points, centroids, N, k, D);
        if (distance < min_distance) {
            min_distance = distance;
            best_cluster = j;
        }
    }

    return best_cluster;
}

void accumulate_sum(
    const int& point,       // Input point coordinates
    const float* points,    // Array of points
    float* new_centroids,   // Array to accumulate sums for centroid updates
    const int best_cluster, // Index of the cluster to which the point is assigned
    const int N,            // Number of points
    const int k,            // Number of centroids
    const int D             // Number of dimensions
    )
{
    for (int d = 0; d < D; d++) {
        new_centroids[d * k + best_cluster] += points[d * N + point];
    }
}

void assign_clusters(
    const float* points,      // Input points array [N x D]
    const float* centroids,   // Centroids array [k x D]
    float* new_centroids,     // Array to accumulate sums for centroid updates [k x D]
    int* assigned_clusters,   // Cluster assignments [N]
    int* cluster_sizes,       // Number of points in each cluster [k]
    int& changed,            // Flag indicating if any assignments changed
    const int N,              // Number of points
    const int D,              // Dimensions
    const int k              // Number of clusters
    )
{
    // For each point find nearest centroids
        for (int i = 0; i < N; i++) {
            int best_cluster = 0;

            // Find closest centroid
            best_cluster = find_closest_centroid(i, points, centroids, N, k, D);
            
            // Update changed flag
            if (assigned_clusters[i] != best_cluster)
                changed++;

            // Assign cluster
            cluster_sizes[assigned_clusters[i]]--;
            assigned_clusters[i] = best_cluster;
            cluster_sizes[best_cluster]++;
            
            // Accumulate sum for centroid update
            accumulate_sum(i, points, new_centroids, best_cluster, N, k, D);
        }
}

void update_centroids(
    const float* new_centroids, // Array to accumulate sums for centroid updates [k x D]
    const int* cluster_sizes,   // Number of points in each cluster [k]
    float* centroids,           // Centroids array [k x D]
    const int D,                // Number of dimensions
    const int k                 // Number of clusters
    )
{
    for (int j = 0; j < k; j++) {
        if (cluster_sizes[j] > 0) {
            for (int d = 0; d < D; d++) {
                centroids[d * k + j] = new_centroids[d * k + j] / cluster_sizes[j];
            }
        }
    }
}
