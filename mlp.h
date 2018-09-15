//
// Created by komesergey on 02/09/2018.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include <vector>
#include "layer.h"

class mlp {
private:
    float learningRate;
    std::vector<layer> layers;
public:
    layer *getLayer(Type type);
    layer* getPrevLayer(Type type);
    layer *createInputLayer(unsigned long inputCount);
    layer *createLayer(unsigned long neuronCount, unsigned long weightCount);
    int getClassification();
    void updateNeuronWeights(Type type, unsigned long id, float error);
    void backPropagateHiddenLayer(int targetClassification);
    void backPropagateOutputLayer(int targetClassification);
    void backPropagate(int targetClassification);
    void activateNeuron(Type type, unsigned long id);
    void calcNeuronOutput(Type type, unsigned long id);
    void calcLayerOutput(Type type);
    void feedForward();
    void feedInput(std::vector<float> *v);
    void init(unsigned long inputCount, unsigned long hiddenCount, unsigned long outputCount);
    void initWeights(Type type);
    void setLearningRate(float learningRate);
    void train(std::vector< std::vector<uint8_t > > trainingImages, std::vector<uint8_t > trainingLabels);
    void test(std::vector< std::vector<uint8_t > > testImages, std::vector<uint8_t > testLabels);
    void displayTrainingProgress(unsigned long imageCount, int errorCount, unsigned long totalCount);
    void displayTestingProgress(unsigned long imageCount, int errorCount, unsigned long totalCount, int classification, int label);
};

#endif //MLP_MLP_H