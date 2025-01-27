#include <stdlib.h>
#include <stdio.h>

#include "mlp.h"
#include "tnet.h"
#include "loss.h"
#include "layer.h"

struct mlp *mlp_init(int numInputs, int numOutputs, int numHiddenLayers, int numParams[numHiddenLayers])
{
    struct mlp *p = calloc(1, sizeof(struct mlp));
    p->numInputs = numInputs;
    p->numOutputs = numOutputs;
    p->lossFn = loss_mse;
    // hidden layers + output layer
    p->layers = calloc(numHiddenLayers + 1, sizeof(struct layer));
    int numWeights = numInputs;
    for (int i = 0; i < numHiddenLayers; i++)
    {
        layer_init(&p->layers[i], numParams[i], numWeights, av_tanh);
        numWeights = numParams[i];
    }
    layer_init(&p->layers[numHiddenLayers], numOutputs, numWeights, av_heaviside);
    return p;
}

vec mlp_forward(struct mlp *p, vec inputs)
{

    vec prev_outputs = inputs;
    for (int i = 0; i < p->numLayers; i++)
    {
        vec n_outputs = layer_forward(&p->layers[i], prev_outputs);
        vec_free(prev_outputs);
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
