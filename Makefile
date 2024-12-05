# CUDA directory
CUDA_ROOT_DIR = /usr/local/cuda

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

clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o

.PHONY: all clean

# # Make file taken and modified from: https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable
# ###########################################################

# ## USER SPECIFIC DIRECTORIES ##

# # CUDA directory:
# CUDA_ROOT_DIR=/usr/local/cuda

# ##########################################################

# ## CC COMPILER OPTIONS ##

# # CC compiler options:
# CC=gcc
# CC_FLAGS=
# CC_LIBS=

# ##########################################################

# ## NVCC COMPILER OPTIONS ##

# # NVCC compiler options:
# NVCC=nvcc
# NVCC_FLAGS= -arch=sm_75 -rdc=true
# NVCC_LIBS=

# # CUDA library directory:
# CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# # CUDA include directory:
# CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# # CUDA linking libraries:
# CUDA_LINK_LIBS= -lcudart

# ##########################################################

# ## Project file structure ##

# # Source file directory:
# SRC_DIR = src

# # Object file directory:
# OBJ_DIR = bin

# # Include header file diretory:
# INC_DIR = include

# ##########################################################

# ## Make variables ##

# # Target executable name:
# EXE = KMeans

# # Object files:
# OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kmeans_cpu.o $(OBJ_DIR)/kmeans_gpu1.o $(OBJ_DIR)/kmeans_gpu2.o $(OBJ_DIR)/utils.o

# ##########################################################

# ## Compile ##

# # Link c++ and CUDA compiled object files to target executable:
# $(EXE) : $(OBJS)
# 	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# # Compile main .cpp file to object files:
# $(OBJ_DIR)/%.o : %.cpp
# 	$(CC) $(CC_FLAGS) -c $< -o $@

# # Compile C++ source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
# 	$(CC) $(CC_FLAGS) -c $< -o $@

# # Compile CUDA source files to object files:
# $(OBJ_DIR)/%.o : %.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# # Compile CUDA source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# # Clean objects in object directory.
# clean:
# 	$(RM) bin/* *.o $(EXE)
