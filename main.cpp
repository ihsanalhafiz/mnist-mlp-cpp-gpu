#include <iostream>
#include "mlp.h"
#include "mnist.h"
#include <chrono>
#include <cuda_runtime.h>

mlp *createNetwork(unsigned long inputCount, unsigned long hiddenCount, unsigned long outputCount){
    auto *nn = new mlp();
    nn->init(inputCount, hiddenCount, outputCount);    // Initialize network (now this should allocate memory on the GPU)
    nn->setLearningRate(0.1);                          // Set learning rate
    nn->initWeights(HIDDEN);                           // Initialize weights on GPU for the hidden layer
    nn->initWeights(OUTPUT);                           // Initialize weights on GPU for the output layer
    return nn;
}

int main() {
    std::cout << "MNIST data directory: " << "data" << std::endl;
    
    // Load dataset
    mnist::dataset dataSet = mnist::readDataSet("data");
    
    // Create and initialize the neural network
    mlp *nn = createNetwork(dataSet.trainingImages.at(0).size(), 80, 10);
    
    // Timing the training process
    auto start = std::chrono::high_resolution_clock::now();
    
    // Train the neural network (this will now run the CUDA version)
    nn->train(dataSet.trainingImages, dataSet.trainingLabels);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto trainingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Training time: " << trainingDuration << " ms" << std::endl;
    
    // Timing the testing process
    start = std::chrono::high_resolution_clock::now();
    
    // Test the neural network (this will now run the CUDA version)
    nn->test(dataSet.testImages, dataSet.testLabels);
    
    stop = std::chrono::high_resolution_clock::now();
    auto testingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Testing time: " << testingDuration << " ms" << std::endl;

    return 0;
}
