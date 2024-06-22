#include <metal_stdlib>
using namespace metal;

kernel void forward_pass(const device float* inputs [[ buffer(0) ]],
                         const device float* weights [[ buffer(1) ]],
                         const device float* biases [[ buffer(2) ]],
                         device float* outputs [[ buffer(3) ]],
                         uint id [[ thread_position_in_grid ]]) {
    // For simplicity, let's assume a single layer with a single output neuron
    // inputs: input features
    // weights: weights of the neuron
    // biases: bias of the neuron
    // outputs: output of the neuron (logistic sigmoid activation)

    float dot_product = 0.0;
    for (int i = 0; i < 2; i++) { // Assume 2 input features for simplicity
        dot_product += inputs[id * 2 + i] * weights[i];
    }
    dot_product += biases[0];
    
    // Sigmoid activation
    outputs[id] = 1.0 / (1.0 + exp(-dot_product));
}
