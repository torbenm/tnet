#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "tnet.h"
#include "mat.h"

void layer_init(struct layer *l, int numNodes, int numInputs, activationfunc *activationFn)
{
    l->numNodes = numNodes;
    l->numInputs = numInputs;
    l->weights = mat_alloc(numNodes, numInputs);
    l->preActivations = vec_alloc(l->numNodes);
    l->activations = vec_alloc(l->numNodes);
    l->inputs = vec_alloc(numInputs);
    l->activationFn = activationFn;
}

vec layer_forward(struct layer *l, vec inputs)
{
    // inputs: inputs x 1
    // l->weights: inputs x nodes
    // preActivations, activations: nodes x 1

    free(l->activations);
    free(l->preActivations);
    l->inputs = inputs;
    l->preActivations = mat_dot_product(l->weights, inputs, l->numNodes, l->numInputs);
    l->activations = l->activationFn(l->preActivations, l->numNodes, ACTIVATION_FORWARD);
    return l->activations;
}

/**
 * previousSmallDelta: 1 x nodes
 */
vec layer_backward(struct layer *l, vec previousSmallDelta)
{

    // l->weights: inputs x nodes
    // weights_t: nodes x inputs
    mat weights_t = mat_transpose(l->weights, l->numNodes, l->numInputs);

    // previousSmallDelta: nodes x 1
    // weights_x_delta: inputs x 1
    vec weights_x_delta = mat_dot_product(weights_t, previousSmallDelta, l->numInputs, l->numNodes);

    // preActivations, activations: nodes x 1
    // actDeriv = nodes x 1
    // newSmallDelta: nodes x 1
    vec actDeriv = l->activationFn(l->preActivations, l->numNodes, ACTIVATION_DERIVATIVE);
    vec newSmallDelta = vec_elem_mul(weights_x_delta, actDeriv, l->numNodes); // this shouldn't work??

    // deltaW: nodes x inputs -- this seems wrong?
    mat deltaW = vec_transposed_vec_mul(newSmallDelta, l->inputs, -1.0, l->numNodes, l->numInputs);
    mat newWeights = mat_elem_add(l->weights, deltaW, l->numNodes, l->numInputs);
    mat_free(l->weights, l->numNodes);
    l->weights = newWeights;

    // vec deltaB = vec_mul_const(newSmallDelta, -1);

    mat_free(weights_t, l->numInputs);
    vec_free(weights_x_delta);
    vec_free(actDeriv);
    mat_free(deltaW, l->numNodes);

    return newSmallDelta;
}

void layer_free(struct layer *l)
{
    mat_free(l->weights, l->numNodes);
    free(l);
}
