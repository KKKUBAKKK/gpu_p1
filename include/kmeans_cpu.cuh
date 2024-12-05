#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

/**
 * @brief Performs k-means clustering algorithm on CPU
 * @param points Input points array [N x D]
 * @param centroids Centroids array [k x D]
 * @param assigned_clusters Cluster assignments [N]
 * @param N Number of points
 * @param D Number of dimensions
 * @param k Number of clusters
 * @param max_iter Maximum iterations
 */
void kmeans_cpu(const float* points, float* centroids, int* assigned_clusters, 
                const int N, const int D, const int k, const int max_iter);

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

/**
 * @brief Finds the closest centroid to a given point using Euclidean distance
 * @param point Index of the point to find closest centroid for
 * @param points Array containing all points coordinates [N x D]
 * @param centroids Array containing all centroids coordinates [k x D]
 * @param N Total number of points in the dataset
 * @param k Total number of centroids/clusters
 * @param D Number of dimensions for each point
 * @return Index of the closest centroid
 */
int find_closest_centroid(const int& point, const float* points, const float* centroids, 
                            const int N, const int k, const int D);

/**
 * @brief Accumulates coordinates sum of a point into its assigned cluster's centroid
 * @param point Index of the point to accumulate
 * @param points Array containing all points coordinates [N x D]
 * @param new_centroids Array to accumulate centroid coordinate sums [k x D]
 * @param best_cluster Index of the cluster to accumulate into
 * @param N Total number of points
 * @param k Number of clusters
 * @param D Number of dimensions
 */
void accumulate_sum(const int& point, const float* points, float* new_centroids, 
                    const int best_cluster, const int N, const int k, const int D);

/**
 * @brief Assigns each point to its nearest cluster and prepares data for centroid updates
 * @param points Input points array containing coordinates [N x D]
 * @param centroids Current centroid positions [k x D]
 * @param new_centroids Array to accumulate centroid coordinate sums [k x D]
 * @param assigned_clusters Array storing cluster assignments for each point [N]
 * @param cluster_sizes Array storing number of points in each cluster [k]
 * @param changed Flag indicating if any point changed its cluster assignment
 * @param N Total number of points in dataset
 * @param D Number of dimensions for each point
 * @param k Number of clusters
 */
void assign_clusters(const float* points, const float* centroids, float* new_centroids, 
                    int* assigned_clusters, int* cluster_sizes, int& changed, 
                    const int N, const int D, const int k);

/**
 * @brief Updates centroids positions by dividing accumulated sums by cluster sizes
 * @param new_centroids Array containing sums for centroid updates [k x D]
 * @param cluster_sizes Array containing number of points in each cluster [k]
 * @param centroids Array containing centroid positions to be updated [k x D]
 * @param D Number of dimensions for each point
 * @param k Number of clusters
 */
void update_centroids(const float* new_centroids, const int* cluster_sizes, float* centroids, 
                        const int D, const int k);

#endif // KMEANS_CPU_H