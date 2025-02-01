#pragma once

#include "core.h"
#include "funcs.h"

struct intermediate
{
    vec activations;
    vec preActivations;
    int nOutputs;
};

struct intermediate *intermediate_alloc(int size);
void intermediate_free(struct intermediate *);

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

struct perceptron
{
    param_t bias;
    vec weights;
    int numWeights;
    activationfunc *activationFn;
};

vec perceptron_forward(struct perceptron *p, vec vals);
int perceptron_train(struct perceptron *p, int maxIter, int numTVals, mat tvals, vec truth, param_t learningRate);
struct perceptron *perceptron_init(int numWeights);
void perceptron_free(struct perceptron *p);
void perceptron_forward_print(struct perceptron *p, int numVals, mat values, vec truth);