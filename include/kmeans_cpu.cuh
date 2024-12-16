#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

/**
 * @brief Calculates Euclidean distance between a point and centroid
 * @param points Array containing all points
 * @param centroids Array containing all centroids
 * @param assigned_clusters Cluster assignments
 * @param N Total number of points
 * @param D Number of dimensions
 * @param k Total number of centroids
 * @param max_iter Maximum number of iterations
 */
void kmeans_cpu(
    const float* points,          // Input points array
    float* centroids,             // Centroids array
    int* assigned_clusters,       // Cluster assignments
    const int N,                  // Number of points
    const int D,                  // Dimensions
    const int k,                  // Number of clusters
    const int max_iter            // Maximum iterations
    );

/**
 * @brief Calculates Euclidean distance between a point and centroid
 * @param point Index of the point
 * @param centroid Index of the centroid
 * @param points Array containing all points
 * @param centroids Array containing all centroids
 * @param N Total number of points
 * @param k Total number of centroids
 * @param D Number of dimensions
 */
float calculate_distance(const int& point, const int& centroid, const float* points, 
                        const float* centroids, const int N, const int k, const int D);
#endif // KMEANS_CPU_H