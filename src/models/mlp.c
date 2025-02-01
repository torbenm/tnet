#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "models.h"
#include "funcs.h"

struct mlp *mlp_init(int numInputs, int numOutputs, int numHiddenLayers, int numParams[numHiddenLayers])
{
    struct mlp *p = malloc(1 * sizeof(struct mlp));
    p->numInputs = numInputs;
    p->numOutputs = numOutputs;
    p->numLayers = numHiddenLayers + 1;
    p->lossFn = loss_mse;
    // hidden layers + output layer
    p->layers = malloc((numHiddenLayers + 1) * sizeof(struct layer *));
    int numWeights = numInputs;
    for (int i = 0; i < numHiddenLayers; i++)
    {
        p->layers[i] = layer_init(numParams[i], numWeights, av_tanh);
        numWeights = numParams[i];
    }
    p->layers[numHiddenLayers] = layer_init(numOutputs, numWeights, av_softmax);
    return p;
}

void mlp_backward(struct mlp *p, vec delta, param_t learningRate, struct intermediate ims[p->numLayers + 1])
{
    vec nextDelta = delta;
    for (int i = p->numLayers - 1; i >= 0; i--)
    {
        nextDelta = layer_backward(p->layers[i], nextDelta, learningRate, &ims[i + 1], &ims[i], (i == (p->numLayers - 1)));
    }
}

vec mlp_forward(struct mlp *p, vec inputs, struct intermediate ims[p->numLayers + 1])
{

    vec prev_outputs = inputs;
    ims[0].activations = inputs;

    for (int i = 0; i < p->numLayers; i++)
    {
        vec n_outputs = layer_forward(p->layers[i], prev_outputs, &ims[i + 1]);
        prev_outputs = n_outputs;
    }
    return prev_outputs;
}

void mlp_free(struct mlp *p)
{
    for (int i = 0; i < p->numLayers; i++)
    {
        layer_free(p->layers[i]);
        p->layers[i] = NULL;
    }
    p->layers = NULL;
    free(p);
}
