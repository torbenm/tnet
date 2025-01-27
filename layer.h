#pragma once

#include "tnet.h"
#include "loss.h"
#include "mat.h"
#include "activation.h"

struct layer
{
    int numNodes;
    int numInputs;
    mat weights;
    activationfunc *activationFn;
};

void layer_init(struct layer *l, int numNodes, int numInputs, activationfunc *activationFn);
void layer_free(struct layer *layer);
param_t *layer_forward(struct layer *l, param_t *values);