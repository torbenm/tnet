#pragma once

#include "tnet.h"
#include "layer.h"
#include "mat.h"
#include "activation.h"
#include "loss.h"

struct mlp
{
    int numLayers;
    int numInputs;
    int numOutputs;
    struct layer **layers;
    lossfunc *lossFn;
};

struct mlp *mlp_init(int numInputs, int numOutputs, int numHiddenLayers, int numParams[numHiddenLayers]);
vec mlp_forward(struct mlp *p, vec inputs, struct intermediate ims[p->numLayers]);
void mlp_backward(struct mlp *p, vec delta, param_t learningRate, struct intermediate ims[p->numLayers]);
