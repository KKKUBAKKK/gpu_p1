#ifndef KMEANS_GPU1_H
#define KMEANS_GPU1_H

/**
 * @brief Structure to hold GPU memory resources
 */
struct GPUResources {
    float *d_points;
    float *d_centroids;
    float *d_new_centroids;
    int *d_assignments;
    int *d_assignments_counter;
    int *d_cluster_sizes;
    float *d_cluster_sums;
    int *d_changed;

    GPUResources() : d_points(nullptr), d_centroids(nullptr), d_new_centroids(nullptr),
                     d_assignments(nullptr), d_assignments_counter(nullptr), d_cluster_sizes(nullptr),
                     d_cluster_sums(nullptr), d_changed(nullptr) {}
};

/**
 * @brief Performs k-means clustering using GPU acceleration (version 1)
 * 
 * @param h_points Pointer to the array of points on the host [N x d]
 * @param h_centroids Pointer to the array of centroids on the host [k x d]
 * @param h_assignments Pointer to the array of cluster assignments on the host [N]
 * @param N Number of points
 * @param d Number of dimensions
 * @param k Number of clusters
 * @param max_iter Maximum number of iterations
 */
void kmeans_gpu1(const float* h_points, float* h_centroids, int* h_assignments, 
                const int N, const int d, const int k, const int max_iter);

/**
 * @brief Cleanup function to free GPU resources
 * @param res Reference to GPUResources structure containing allocated memory
 */
void cleanup_gpu_resources(GPUResources& res);

#endif // KMEANS_GPU1_H