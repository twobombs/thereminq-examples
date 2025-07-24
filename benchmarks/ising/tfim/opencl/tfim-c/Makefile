# Makefile for compiling the OpenCL Ising Sampler

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -std=c99 -Wall -O3

# Linker flags
# For NVIDIA/AMD/Intel on Linux: -lOpenCL
# For macOS: -framework OpenCL
LDFLAGS = -lOpenCL -lm

# Target executable
TARGET = ising_sampler_opencl

# Source file
SRC = ising_sampler_opencl.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
