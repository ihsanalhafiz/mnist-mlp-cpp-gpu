#include <iostream>
#include "mlp.h"
#include "mnist.h"
#include <chrono>

mlp *createNetwork(unsigned long inputCount, unsigned long hiddenCount, unsigned long outputCount){
    auto *nn = new mlp();
    nn->init(inputCount, hiddenCount, outputCount);
    nn->setLearningRate(0.2);
    nn->initWeights(HIDDEN);
    nn->initWeights(OUTPUT);
    return nn;
}

int main() {
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    mnist::dataset dataSet = mnist::readDataSet(MNIST_DATA_LOCATION);
    mlp *nn = createNetwork(dataSet.trainingImages.at(0).size(), 20, 10);
    nn->train(dataSet.trainingImages, dataSet.trainingLabels);
    nn->test(dataSet.testImages, dataSet.testLabels);
    return 0;
}