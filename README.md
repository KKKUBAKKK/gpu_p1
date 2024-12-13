# K-Means algorithm with one CPU version and two GPU versions

## Usage

### Compile and run:
```bash
make clean
make
./KMeans <data_format> <computation_mode> <input_file> <output_file>
```

### Parameters:
data_format:     txt | bin

computation_mode: cpu | gpu1 | gpu2

input_file:      Path to input data file

output_file:     Path to save results

## CPU Implementation

### Overview
Sequential CPU implementation of K-means clustering that serves as a baseline for performance comparison.

### Algorithm
```python
while (iterations < max_iter AND points_changed > 0):
    1. Reset cluster counters and centroids
    2. For each point:
        - Find closest centroid (Euclidean distance)
        - Assign point to closest cluster
        - Update running sum for new centroid position
    3. For each cluster:
        - Calculate new centroid position (average)
    4. Count points that changed clusters
```

## GPU Implementation (Version 1)

### Overview
First GPU implementation of K-means clustering utilizing CUDA for parallel computation across multiple thread blocks.

### Algorithm Structure
```python
Host:
1. Allocate and transfer data to GPU
2. For each iteration:
    - Find nearest centroids (parallel)
    - Sum points for each cluster (parallel)
    - Update centroid positions (parallel)
3. Transfer results back to host

Device (Kernels):
1. findNearestCentroids:
    - Each thread processes one point
    - Compute distances to all centroids
    - Atomic updates for changed assignments

2. sum:
    - Parallel reduction using shared memory
    - Each block processes subset of points
    - Atomic updates for cluster sums

3. update:
    - Single block updates all centroids
    - Division by cluster sizes
```

## GPU Implementation (Version 2)

### Overview
Second GPU implementation of K-means clustering focusing on optimized memory access and reduced kernel launches.

### Algorithm Structure
```python
Single Kernel Approach:
1. Load centroids into shared memory
2. Each thread:
    - Process one point
    - Find nearest centroid
    - Update local cluster sums
3. Reduce cluster sums across blocks
4. Update centroids in-place
```