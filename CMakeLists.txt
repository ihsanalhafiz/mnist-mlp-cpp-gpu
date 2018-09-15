cmake_minimum_required(VERSION 3.0)

PROJECT(mnist-mlp-cpp)

add_executable(mnist_example main.cpp mlp.cpp mlp.h neuron.cpp neuron.h layer.cpp layer.h mnist.h)
target_compile_features(mnist_example PRIVATE cxx_range_for)

target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${PROJECT_SOURCE_DIR}/data")
