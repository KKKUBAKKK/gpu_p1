#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>
#include "include/utils.cuh"
#include "include/kmeans_cpu.cuh"
#include "include/kmeans_gpu1.cuh"
#include "include/kmeans_gpu2.cuh"
#include <iostream>

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
    std::cout << "Starting k-means clustering program" << std::endl;

    // Read the command line input
    read_command_line_input(argv, argc, input_path, output_path, format, mode);
    std::cout << "Input path: " << input_path << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    std::cout << "Data format: " << (format == TEXT ? "TEXT" : "BINARY") << std::endl;
    std::cout << "Computation mode: " << (mode == CPU ? "CPU" : mode == GPU1 ? "GPU1" : "GPU2") << std::endl;

    // 2nd step: Loading data
    if (format == TEXT)
    {
        std::cout << "Loading text data from " << input_path << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        load_text_data(input_path, N, D, k, points);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Data loaded successfully in time: " << duration.count() << " ms" << std::endl;
    }
    else
    {
        std::cout << "Loading binary data from " << input_path << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        load_binary_data(input_path, N, D, k, points);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Data loaded successfully in time: " << duration.count() << " ms" << std::endl;
    }

    // Printing basic information about the data
    std::cout << "Number of points: " << N << std::endl;
    std::cout << "Number of dimensions: " << D << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Maximum number of iterations: " << MAX_ITERATIONS << std::endl;

    // Initialize the centroids
    initialize_centroids(points, centroids, N, D, k);

    // Allocate memory for the cluster assignments
    try {
        assigned_clusters = new int[N];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for assigned_clusters: " << e.what() << std::endl;
        exit(1);
    }

    // 3rd step: Starting the k-means clustering algorithm
    if (mode == CPU)
    {
        std::cout << "Starting the CPU version of the k-means clustering algorithm" << std::endl;

        kmeans_cpu(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        std::cout << "K-means clustering completed" << std::endl;
    }
    else if (mode == GPU1)
    {
        std::cout << "Starting the GPU1 version of the k-means clustering algorithm" << std::endl;

        kmeans_gpu1(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        std::cout << "K-means clustering completed" << std::endl;
    }
    else
    {
        std::cout << "Starting the GPU2 version of the k-means clustering algorithm" << std::endl;

        kmeans_gpu2(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        std::cout << "K-means clustering completed" << std::endl;
    }

    // 4th step: Saving the results
    std::cout << "Starting saving the results to " << output_path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    save_text_results(output_path, N, D, k, assigned_clusters, centroids);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Results saved successfully in time: " << duration.count() << " ms" << std::endl;

    // Free the memory
    delete[] points;
    delete[] centroids;
    delete[] assigned_clusters;

    return 0;
}