#ifndef KMEANS_GPU2_H
#define KMEANS_GPU2_H

/**
 * @brief Performs k-means clustering using GPU acceleration (version 2)
 * 
 * @param h_points Pointer to the array of points on the host [N x d]
 * @param h_centroids Pointer to the array of centroids on the host [k x d]
 * @param h_assignments Pointer to the array of cluster assignments on the host [N]
 * @param N Number of points
 * @param d Number of dimensions
 * @param k Number of clusters
 * @param max_iter Maximum number of iterations
 */
void kmeans_gpu2(const float* h_points, float* h_centroids, int* h_assignments, 
                const int N, const int d, const int k, const int max_iter);

#endif // KMEANS_GPU2_H