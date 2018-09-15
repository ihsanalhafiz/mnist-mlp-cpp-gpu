//
// Created by komesergey on 02/09/2018.
//

#ifndef MLP_NEURON_H
#define MLP_NEURON_H
#include <vector>

class neuron {
    private:
        float bias;
        float output;
        std::vector<float> weights;
    public:
        std::vector<float>* getWeights();
        void setWeights(std::vector<float>* weights);
        float getOutput();
        void setOutput(float output);
        float getBias();
        void setBias(float bias);
};

#endif //MLP_NEURON_H
