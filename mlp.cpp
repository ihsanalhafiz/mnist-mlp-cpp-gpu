//
// Created by komesergey on 02/09/2018.
//

#include "mlp.h"
#include <limits>
#include <iostream>
#include <cmath>
#include <random>

layer * mlp::getLayer(Type type){
    return &layers[type];
}
float mlp::getDerivative(float outVal){
    return outVal * (1-outVal);
}

void mlp::updateNeuronWeights(Type type, unsigned long id, float error){
    neuron *updateNeuron = getLayer(type)->getNeuron(id);
    for (unsigned long i = 0; i < updateNeuron->getWeights()->size(); i++){
        updateNeuron->getWeights()->at(i) += (this->learningRate * getPrevLayer(type)->getNeuron(i)->getOutput() * error);
    }
    updateNeuron->setBias(updateNeuron->getBias() + (this->learningRate * 1 * error));
}

void mlp::backPropagateHiddenLayer(int targetClassification){
    layer *outputLayer = getLayer(OUTPUT);
    for (unsigned long h = 0;h < getLayer(HIDDEN)->getNeurons()->size(); h++){
        float outputCellErrorSum = 0;
        for (unsigned long o = 0;o < outputLayer->getNeurons()->size(); o++){
            neuron *on = outputLayer->getNeuron(o);
            int targetOutput = (o == targetClassification) ? 1 : 0;
            float errorDelta = targetOutput - on->getOutput();
            float errorSignal = errorDelta * getDerivative(on->getOutput());
            outputCellErrorSum += errorSignal * on->getWeights()->at(h);
        }
        float hiddenErrorSignal = outputCellErrorSum * getDerivative(getLayer(HIDDEN)->getNeuron(h)->getOutput());
        updateNeuronWeights(HIDDEN, h, hiddenErrorSignal);
    }
}

void mlp::backPropagateOutputLayer(int targetClassification){
    for (unsigned long o = 0;o < getLayer(OUTPUT)->getNeurons()->size(); o++){
        neuron *on = getLayer(OUTPUT)->getNeuron(o);
        int targetOutput = (o == targetClassification) ? 1 : 0;
        float errorDelta = targetOutput - on->getOutput();
        float errorSignal = errorDelta * getDerivative(on->getOutput());
        updateNeuronWeights(OUTPUT, o, errorSignal);
    }
}

void mlp::backPropagate(int targetClassification){
    backPropagateOutputLayer(targetClassification);
    backPropagateHiddenLayer(targetClassification);
}

void mlp::activateNeuron(Type type, unsigned long id){
    neuron *n = getLayer(type)->getNeuron(id);
    n->setOutput(1 / (1 + (exp(-n->getOutput())) ));
}

layer* mlp::getPrevLayer(Type type){
    return type == HIDDEN ? getLayer(INPUT) : getLayer(HIDDEN);
}

void mlp::calcNeuronOutput(Type type, unsigned long id){
    neuron *calcNeuron = getLayer(type)->getNeuron(id);
    calcNeuron->setOutput(calcNeuron->getBias());
    for (unsigned long i = 0; i < getPrevLayer(type)->getNeurons()->size(); i++){
        calcNeuron->setOutput(calcNeuron->getOutput() + getPrevLayer(type)->getNeuron(i)->getOutput() * calcNeuron->getWeights()->at(i));
    }
}

void mlp::calcLayerOutput(Type type){
    for (unsigned long i = 0;i < getLayer(type)->getNeurons()->size(); i++){
        calcNeuronOutput(type, i);
        activateNeuron(type,i);
    }
}

void mlp::feedForward(){
    calcLayerOutput(HIDDEN);
    calcLayerOutput(OUTPUT);
}

void mlp::feedInput(std::vector<float> *v) {
    for (unsigned long i = 0; i < v->size(); i++){
        neuron * iln = getLayer(INPUT)->getNeuron(i);
        iln->setOutput(v->at(i));
    }
}

layer *mlp::createInputLayer(unsigned long inputCount){
    auto *il = new layer();
    il->setNeurons(new std::vector<neuron>(inputCount));
    return il;
}

layer *mlp::createLayer(unsigned long neuronCount, unsigned long weightCount){
    auto *l = new layer();
    auto layerNeurons = new std::vector<neuron>();
    layerNeurons->reserve(neuronCount);
    for (int i = 0;i < neuronCount; i++){
        auto *dn = new neuron();
        dn->setBias(0.0f);
        dn->setOutput(0.0f);
        dn->setWeights(new std::vector<float>(weightCount));
        layerNeurons->push_back(*dn);
    }
    l->setNeurons(layerNeurons);
    return l;
}

void mlp::init(unsigned long inputCount, unsigned long hiddenCount, unsigned long outputCount){
    layers = *new std::vector<layer>(3);
    layers[0] = *createInputLayer(inputCount);
    layers[1] = *createLayer(hiddenCount, inputCount);
    layers[2] = *createLayer(outputCount, hiddenCount);
}

void mlp::setLearningRate(float learningRate) { this->learningRate = learningRate; }

void mlp::initWeights(Type type){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    layer *l = getLayer(type);
    for (unsigned long o=0; o<l->getNeurons()->size();o++){
        neuron *n = l->getNeuron(o);
        for (unsigned long i = 0; i < n->getWeights()->size(); i++){
            n->getWeights()->at(i) = 0.7f*(dis(gen));
            if (i%2) n->getWeights()->at(i) = -n->getWeights()->at(i);
        }
        n->setBias(dis(gen));
        if (o%2) n->setBias(-n->getBias());
    }
}

void mlp::train(std::vector< std::vector<uint8_t > > trainingImages, std::vector<uint8_t > trainingLabels){
    std::cout << "Number of training images = " << trainingImages.size() << std::endl;
    std::cout << "Number of training labels = " << trainingLabels.size() << std::endl;
    int errCount = 0;
    for (unsigned long imgCount=0; imgCount < trainingImages.size(); imgCount++){
        std::vector<uint8_t > image = trainingImages[imgCount];
        int label = trainingLabels[imgCount];
        auto normalizedImage = new std::vector< float >(trainingImages[imgCount].size());
        for(unsigned long j =0; j < image.size(); j++){
            normalizedImage->at(j) = image[j] ? 1 : 0;
        }
        this->feedInput(normalizedImage);
        this->feedForward();
        this->backPropagate(label);
        int classification = getClassification();
        if (classification!=label) errCount++;
        displayTrainingProgress(imgCount, errCount, trainingImages.size());
    }
}

void mlp::test(std::vector< std::vector<uint8_t > > testImages, std::vector<uint8_t > testLabels){
    std::cout << "Number of test images = " << testImages.size() << std::endl;
    std::cout << "Number of test labels = " << testLabels.size() << std::endl;
    int errCount = 0;
    for (unsigned long imgCount = 0; imgCount < testImages.size(); imgCount++){
        std::vector<uint8_t > image = testImages[imgCount];
        int label = testLabels[imgCount];
        auto normalizedImage = new std::vector< float >(image.size());
        for(unsigned long j =0; j < image.size(); j++){
            normalizedImage->at(j) = image[j] ? 1 : 0;
        }
        this->feedInput(normalizedImage);
        this->feedForward();
        int classification = getClassification();
        if (classification!=label) {
            errCount++;
            std::cout << "PREDICTED: " << classification << " ACTUAL: " << label << std::endl;
        }
        displayTestingProgress(imgCount, errCount, testImages.size());
    }
}

int mlp::getClassification(){
    layer *l = getLayer(OUTPUT);
    double maxOut = 0;
    unsigned long maxInd = 0;
    for (unsigned long i = 0; i < l->getNeurons()->size(); i++){
        neuron *on = l->getNeuron(i);
        if (on->getOutput() > maxOut){
            maxOut = on->getOutput();
            maxInd = i;
        }
    }
    return (int)maxInd;
}

void mlp::displayTrainingProgress(unsigned long imageCount, int errorCount, unsigned long totalCount){
    double progress = (double)(imageCount + 1)/(double)(totalCount) * 100;
    printf("1: TRAINING: reading image %5ld / %5ld progress [%3d%%]  ",(imageCount + 1),totalCount,(int)progress);
    double accuracy = (1 - ((double)errorCount/(double)(imageCount + 1))) * 100;
    printf("RESULTS: correct=%5ld  incorrect=%5d  accuracy=%5.4f%% \n",imageCount + 1 - errorCount, errorCount, accuracy);
}

void mlp::displayTestingProgress(unsigned long imageCount, int errorCount, unsigned long totalCount){
    double progress = (double)(imageCount + 1)/(double)(totalCount) * 100;
    printf("2: TESTING:  reading image %5ld / %5ld progress [%3d%%]  ",(imageCount + 1),totalCount,(int)progress);
    double accuracy = (1 - ((double)errorCount/(double)(imageCount + 1))) * 100;
    printf("TOTAL: correct=%5ld  incorrect=%5d  accuracy=%5.4f%% \n",imageCount + 1 - errorCount, errorCount, accuracy);
}