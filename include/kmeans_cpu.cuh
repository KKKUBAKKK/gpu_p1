#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

void kmeans_cpu(
    const float* points,          // Input points array [N x D]
    float* centroids,             // Centroids array [k x D]
    int* assigned_clusters,       // Cluster assignments [N]
    const int N,                  // Number of points
    const int D,                  // Dimensions
    const int k,                  // Number of clusters
    const int max_iter            // Maximum iterations
    );