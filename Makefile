NVCC = nvcc
CFLAGS = -Iinclude
CUDAFLAGS = -Iinclude --compiler-options '-fPIC' -ccbin gcc

all: kmeans

kmeans: main.cu src/kmeans_cpu.cu src/utils.cu
	$(NVCC) $(CUDAFLAGS) -o kmeans main.cu src/kmeans_cpu.cu src/utils.cu

clean:
	rm -f kmean