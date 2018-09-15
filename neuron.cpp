//
// Created by komesergey on 02/09/2018.
//

#include <cstdlib>
#include <cmath>
#include "neuron.h"

float neuron::getBias() { return bias; }
void neuron::setBias(float bias) { this->bias = bias; }
float neuron::getOutput() { return output; }
void neuron::setOutput(float output) {this->output = output;}
void neuron::setWeights(std::vector<float> *weights) {this->weights = *weights; }
std::vector<float>* neuron::getWeights() { return &weights; }