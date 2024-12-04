#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>
#include "utils.h"
#include "kmeans_cpu.h"

#define MAX_ITERATIONS 100

int main(int argc, char** argv)
{
    // Initialize variables
    int N, D, k;
    float *points = nullptr, *centroids = nullptr;
    int *assigned_clusters = nullptr;
    std::string input_path, output_path;
    Format format;
    Mode mode;

    // 1st step: Starting program
    printf("Starting k-means clustering program\n");

    // Read the command line input
    read_command_line_input(argv, argc, input_path, output_path, format, mode);
    printf("Input path: %s\n", input_path);
    printf("Output path: %s\n", output_path);
    printf("Data format: %s\n", format == TEXT ? "TEXT" : "BINARY");
    printf("Computation mode: %s\n", mode == CPU ? "CPU" : mode == GPU1 ? "GPU1" : "GPU2");

    // 2nd step: Loading data
    if (format == TEXT)
    {
        printf("Loading text data from %s\n", input_path);
        auto start_time = std::chrono::high_resolution_clock::now();

        load_text_data(input_path, N, D, k, points);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("Data loaded successfully in time: %ld ms\n", duration.count());
    }
    else
    {
        printf("Loading binary data from %s\n", input_path);
        auto start_time = std::chrono::high_resolution_clock::now();

        load_binary_data(input_path, N, D, k, points);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("Data loaded successfullyin time: %ld ms\n", duration.count());
    }

    // Printing basic information about the data
    printf("Number of points: %d\n", N);
    printf("Number of dimensions: %d\n", D);
    printf("Number of clusters: %d\n", k);
    printf("Maximum number of iterations: %d\n", MAX_ITERATIONS);

    // Initialize the centroids
    initialize_centroids(points, centroids, N, D, k);

    // Allocate memory for the cluster assignments
    try {
        assigned_clusters = new int[N];
    } catch (const std::bad_alloc& e) {
        fprintf(stderr, "Error: Unable to allocate memory for assigned_clusters: %s\n", e.what());
        exit(1);
    }

    // 3rd step: Starting the k-means clustering algorithm
    if (mode == CPU)
    {
        printf("Starting the CPU version of the k-means clustering algorithm\n");

        kmeans_cpu(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        printf("K-means clustering completed\n");
    }
    else if (mode == GPU1)
    {
        printf("Starting the GPU1 version of the k-means clustering algorithm\n");

        // kmeans_gpu1(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        printf("K-means clustering completed\n");
    }
    else
    {
        printf("Starting the GPU2 version of the k-means clustering algorithm\n");

        // kmeans_gpu2(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        printf("K-means clustering completed\n");
    }

    // 4th step: Saving the results
    printf("Starting saving the results to %s\n", output_path);
    auto start = std::chrono::high_resolution_clock::now();

    save_text_results(output_path, N, D, k, assigned_clusters, centroids);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Results saved successfully in time: %ld ms\n", duration.count());

    // Free the memory
    delete[] points;
    delete[] centroids;
    delete[] assigned_clusters;

    return 0;
}