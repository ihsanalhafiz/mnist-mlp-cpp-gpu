# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g -O3 -std=c++17 -I/usr/local/include -L/usr/local/lib 

#include directories src
CXXFLAGS += -Isrc

# Source files and headers
SRCS = src/layer.cpp src/mlp.cpp src/neuron.cpp main.cpp
HEADERS = src/layer.h src/mlp.h src/neuron.h src/mnist.h

# Output binary name
TARGET = main_executable

# Rule to compile the project
all: clean $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Rule to clean the compiled files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
