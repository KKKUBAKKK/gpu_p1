# Directory structure
SRC_DIR = src
INC_DIR = include
OBJ_DIR = bin

# NVCC configuration
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -rdc=true -std=c++11 -Xcompiler "-Wall" -ccbin gcc
CUDA_LIBS = -lcudart

# Target executable
TARGET = KMeans

# Source files
SOURCES = main.cu \
	$(SRC_DIR)/kmeans_cpu.cu \
	$(SRC_DIR)/kmeans_gpu1.cu \
	$(SRC_DIR)/kmeans_gpu2.cu \
	$(SRC_DIR)/utils.cu

# Object files
OBJECTS = $(OBJ_DIR)/main.o \
	$(OBJ_DIR)/kmeans_cpu.o \
	$(OBJ_DIR)/kmeans_gpu1.o \
	$(OBJ_DIR)/kmeans_gpu2.o \
	$(OBJ_DIR)/utils.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -I$(CUDA_ROOT_DIR)/include -L$(CUDA_ROOT_DIR)/lib64 $(CUDA_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o

.PHONY: all clean