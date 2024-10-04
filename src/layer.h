//
// Created by komesergey 02/09/2018.
//

#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <vector>
#include "neuron.h"

typedef enum Type {INPUT, HIDDEN, OUTPUT} Type;

class layer {
    private:
        std::vector<neuron> neurons;
    public:
        std::vector<neuron>* getNeurons();
        void setNeurons(std::vector<neuron>* neurons);
        neuron *getNeuron(unsigned long id);
};


#endif //MLP_LAYER_H
