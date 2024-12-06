#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <iostream>

#define MAX_ITERATIONS 100

enum Mode {
    CPU,
    GPU1,
    GPU2
};

enum Format {
    TEXT,
    BINARY
};

// Function to load data from a text file
void load_text_data(
    const std::string path, // Path to the file
    int& N,           // Number of points
    int& D,           // Dimension of the points
    int& k,           // Number of clusters
    float*& points    // Output points
    )
{
    // Define the variables
    FILE* file;
    int result;

    // Read the file using C file I/O
    file = fopen(path.c_str(), "r");
    if (file == NULL)
    {
        std::cerr << "Error: Unable to open the file" << std::endl;
        exit(1);
    }

    // Read the number of points, dimension and clusters
    result = fscanf(file, "%d %d %d", &N, &D, &k);
    if (result != 3)
    {
        std::cerr << "Error: Unable to read the number of points, dimension and clusters" << std::endl;
        exit(1);
    }

    // Allocate memory for the points
    try {
        points = new float[N * D];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for the points: " << e.what() << std::endl;
        exit(1);
    }

    // Read the points
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fscanf(file, "%f", &points[j * N + i]);
        }
    }
}

// Function to load data from a binary file
void load_binary_data(
    const std::string path, // Path to the file
    int& N,           // Number of points
    int& D,           // Dimension of the points
    int& k,           // Number of clusters
    float*& points   // Output points
    )
{
    // Define the variables
    FILE* file;
    int result;

    // Read the file using C file I/O
    file = fopen(path.c_str(), "rb");
    if (file == NULL)
    {
        std::cerr << "Error: Unable to open the file" << std::endl;
        exit(1);
    }

    // Read the number of points, dimension and clusters
    result = fread(&N, sizeof(int), 1, file);
    if (result != 1)
    {
        std::cerr << "Error: Unable to read the number of points" << std::endl;
        exit(1);
    }
    result = fread(&D, sizeof(int), 1, file);
    if (result != 1)
    {
        std::cerr << "Error: Unable to read the dimension of the points" << std::endl;
        exit(1);
    }
    result = fread(&k, sizeof(int), 1, file);
    if (result != 1)
    {
        std::cerr << "Error: Unable to read the number of clusters" << std::endl;
        exit(1);
    }

    // Allocate memory for the points
    try {
        points = new float[N * D];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for the points: " << e.what() << std::endl;
        exit(1);
    }

    // Read the points
    float temp[D];
    size_t elements_read;
    int point = 0;
    while ((elements_read = fread(temp, sizeof(float), D, file)) > 0) 
    {
        if (elements_read != D) {
            if (feof(file)) {
                // Normal EOF reached
                break;
            }
            if (ferror(file)) {
                std::cerr << "Error reading file" << std::endl;
                exit(1);
            }
        }

        for (int i = 0; i < D; i++) {
            points[i * N + point] = temp[i];
        }

        point++;
    }
}

// Function to save results to a text file
void save_text_results(
    const std::string path,                // Path to the file
    const int& N,                    // Number of points
    const int& D,                    // Dimension of the points
    const int& k,                    // Number of clusters
    const int* assigned_clusters,    // Assigned clusters
    const float* centroids           // Final centroids
    )
{
    // Define the variables
    FILE* file;

    // Write the file using C file I/O
    file = fopen(path.c_str(), "w");
    if (file == NULL)
    {
        std::cerr << "Error: Unable to open the file" << std::endl;
        exit(1);
    }

    // Write the centroids
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fprintf(file, "%10f", centroids[j * k + i]);
        }
        fprintf(file, "\n");
    }

    // Write the assigned clusters
    for (int i = 0; i < N; i++)
    {
        fprintf(file, "%3d\n", assigned_clusters[i]);
    }
}

// Function to compare two results files
bool compare_results(
    const std::string path1, // Path to the first file
    const std::string path2  // Path to the second file
    )
{
    // Define the variables
    FILE* file1;
    FILE* file2;
    int result1;
    int result2;
    char line1[256];
    char line2[256];

    // Read the files using C file I/O
    file1 = fopen(path1.c_str(), "r");
    if (file1 == NULL)
    {
        std::cerr << "Error: Unable to open the first file" << std::endl;
        exit(1);
    }
    file2 = fopen(path2.c_str(), "r");
    if (file2 == NULL)
    {
        std::cerr << "Error: Unable to open the second file" << std::endl;
        exit(1);
    }

    // Compare the files
    while (true)
    {
        result1 = fscanf(file1, "%s", line1);
        result2 = fscanf(file2, "%s", line2);
        if (result1 != result2)
        {
            return false;
        }
        if (result1 == EOF)
        {
            break;
        }
        if (strcmp(line1, line2) != 0)
        {
            return false;
        }
    }

    return true;
}

// Function to read command line input
void read_command_line_input(
    char** argv,           // Command line arguments
    const int& argc,       // Number of arguments
    std::string& input_path,     // Path to the input file
    std::string& output_path,    // Path to the output file
    Format& format,        // Data format
    Mode& mode             // Computation mode
    )
{
    // Check the number of arguments
    if (argc != 5)
    {
        std::cerr << "Error: \n \t Usage: KMeans data_format computation_method input_file output_file\n";
        exit(1);
    }

    // Check the data format
    if (strcmp(argv[1], "txt") == 0)
    {
        format = TEXT;
    }
    else if (strcmp(argv[1], "bin") == 0)
    {
        format = BINARY;
    }
    else
    {
        std::cerr << "Error: Invalid data format (txt or bin)" << std::endl;
        exit(1);
    }

    // Check the computation method
    if (strcmp(argv[2], "cpu") == 0)
    {
        mode = CPU;
    }
    else if (strcmp(argv[2], "gpu1") == 0)
    {
        mode = GPU1;
    }
    else if (strcmp(argv[2], "gpu2") == 0)
    {
        mode = GPU2;
    }
    else
    {
        std::cerr << "Error: Invalid computation method (cpu, gpu1 or gpu2)" << std::endl;
        exit(1);
    }

    // Allocate memory for the paths
    input_path = std::string(argv[3]);
    output_path = std::string(argv[4]);
}

// Funtion to initialize centroids
void initialize_centroids(
    const float* points,          // Input points
    float*& centroids,             // Output centroids
    const int N,                  // Number of points
    const int D,                  // Dimension of the points
    const int k                   // Number of clusters
    )
{
    // Allocate memory for the cluster sizes
    try {
        centroids = new float[k * D];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Unable to allocate memory for centroids: " << e.what() << std::endl;
        exit(1);
    }
 
    // Initialize centroids with first k points
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < D; j++)
        {
            centroids[j * k + i] = points[j * N + i];
        }
    }
}

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

        // kmeans_cpu(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        std::cout << "K-means clustering completed" << std::endl;
    }
    else if (mode == GPU1)
    {
        std::cout << "Starting the GPU1 version of the k-means clustering algorithm" << std::endl;

        // kmeans_gpu1(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

        std::cout << "K-means clustering completed" << std::endl;
    }
    else
    {
        std::cout << "Starting the GPU2 version of the k-means clustering algorithm" << std::endl;

        // kmeans_gpu2(points, centroids, assigned_clusters, N, D, k, MAX_ITERATIONS);

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