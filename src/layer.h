#pragma once

#include "tnet.h"
#include "loss.h"
#include "mat.h"
#include "activation.h"
#include "intermediate.h"

struct layer
{
    int numNodes;
    int numInputs;
    mat weights;
    vec bias;
    activationfunc *activationFn;
};

struct layer *layer_init(int numNodes, int numInputs, activationfunc *activationFn);
void layer_free(struct layer *layer);
vec layer_forward(struct layer *l, vec inputs, struct intermediate *i);
vec layer_backward(struct layer *l, vec previousSmallDelta, param_t learningRate, struct intermediate *curr, struct intermediate *prev, int isOutputLayer);