#ifndef UTILS_H
#define UTILS_H

#include <string>

/** @brief Loads data from a text file
 *  @param path Path to the input file
 *  @param N Number of points (output parameter)
 *  @param D Dimension of points (output parameter)
 *  @param k Number of clusters (output parameter)
 *  @param points Array of points (output parameter)
 */
void load_text_data(const std::string path, int& N, int& D, int& k, float*& points);

/** @brief Loads data from a binary file
 *  @param path Path to the input file
 *  @param N Number of points (output parameter)
 *  @param D Dimension of points (output parameter)
 *  @param k Number of clusters (output parameter)
 *  @param points Array of points (output parameter)
 */
void load_binary_data(const std::string path, int& N, int& D, int& k, float*& points);

/** @brief Saves clustering results to a text file
 *  @param path Path to the output file
 *  @param N Number of points
 *  @param D Dimension of points
 *  @param k Number of clusters
 *  @param assigned_clusters Array containing cluster assignments for each point
 *  @param centroids Array containing final centroid positions
 */
void save_text_results(const std::string path, const int& N, const int& D, const int& k, const int* assigned_clusters, const float* centroids);

/** @brief Compares two result files for equality
 *  @param path1 Path to the first result file
 *  @param path2 Path to the second result file
 *  @return True if files are equal, false otherwise
 */
bool compare_results(const std::string path1, const std::string path2);

/** @brief Enumeration defining available computation modes for k-means clustering
 *  CPU: Sequential CPU implementation
 *  GPU1: First GPU implementation
 *  GPU2: Second GPU implementation
 */
enum Mode {
    CPU,
    GPU1,
    GPU2
};

/** @brief Enumeration defining supported data file formats
 *  TEXT: Text-based file format (.txt)
 *  BINARY: Binary file format (.bin)
 */
enum Format {
    TEXT,
    BINARY
};

/** @brief Parses command line arguments to extract input/output paths and execution parameters
 *  @param argv Command line arguments array
 *  @param argc Number of command line arguments
 *  @param input_path Path to input data file (output parameter)
 *  @param output_path Path to output results file (output parameter)
 *  @param format Data file format (output parameter)
 *  @param mode Computation mode (output parameter)
 */
void read_command_line_input(char** argv, const int& argc, std::string& input_path, std::string& output_path, Format& format, Mode& mode);

/** @brief Initializes k centroids by randomly selecting k points from the input dataset
 *  @param points Input array containing N points in D-dimensional space
 *  @param centroids Output array to store the k selected centroids
 *  @param N Number of points in the dataset 
 *  @param D Dimension of each point
 *  @param k Number of clusters/centroids to select
 */
void initialize_centroids(const float* points, float*& centroids, int N, int D, int k);

#endif // UTILS_H