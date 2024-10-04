//
// Created by komesergey on 02/09/2018.
//

#include "layer.h"

std::vector<neuron>* layer::getNeurons() { return &neurons; }
void layer::setNeurons(std::vector<neuron> *neurons) { this->neurons = *neurons;}
neuron* layer::getNeuron(unsigned long id) { return &neurons[id]; }