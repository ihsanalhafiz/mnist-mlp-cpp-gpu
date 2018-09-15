//
// Created by komesergey on 02/09/2018.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include <vector>
#include "layer.h"
#include "mnist.h"

class mlp {
    private:
        float learningRate;
        std::vector<layer> layers;
    public:
        layer *getLayer(Type type);
        layer* getPrevLayer(Type type);
        layer *createInputLayer(unsigned long inputCount);
        layer *createLayer(unsigned long neuronCount, unsigned long weightCount);
        float getDerivative(float outVal);
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
        void train(mnist::dataset dataSet);
        void test(mnist::dataset dataSet);
        void displayTrainingProgress(unsigned long imageCount, int errorCount, int y, int x, unsigned long totalCount);
        void displayTestingProgress(unsigned long imageCount, int errorCount, int y, int x, unsigned long totalCount);
        void locateCursor(int row, int col);
};

#endif //MLP_MLP_H
