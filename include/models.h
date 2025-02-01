#pragma once

#include "core.h"
#include "funcs.h"

struct forwardstate
{
    vec activations;
    vec preActivations;
    int nOutputs;
};

struct forwardstate *forwardstate_alloc(int size);
void forwardstate_free(struct forwardstate *);

struct backwardstate
{
    int numNodes;
    int numInputs;
    mat weightGradients;
    vec biasGradients;
    vec smallDelta;
};

struct backwardstate *backwardstate_alloc(int numNodes, int numInputs);
void backwardstate_free(struct backwardstate *);
void backwardstate_incorporate(struct backwardstate *dst, struct backwardstate *src, param_t factor);

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

/**
 * NEW Seqmodel & its possible models
 */
#define SEQMODEL_STD_SIZE 4

typedef vec seqmodel_layer_forward(void *layer_struct, vec input, struct forwardstate *state);
typedef struct backwardstate *seqmodel_layer_backward(void *p, vec previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev, param_t learningRate);
typedef void seqmodel_layer_update_weights(void *p, struct backwardstate *bs, param_t updateFactor);
struct seqmodel_layer
{
    void *layerProps;
    int numOutputs;
    seqmodel_layer_forward *forward;
    seqmodel_layer_backward *backward;
    seqmodel_layer_update_weights *update;
};

struct seqmodel
{
    int numLayers;
    int _layerBufferSize;
    struct seqmodel_layer **layers;
};
struct seqmodel *seqmodel_init();
void seqmodel_resize(struct seqmodel *seq, int newSize);
void seqmodel_push(struct seqmodel *seq, struct seqmodel_layer *layer);
vec seqmodel_predict(struct seqmodel *seq, vec input);

struct denselayer_props
{
    int numNodes;
    int numInputs;
    mat weights;
    vec bias;
    activationfunc *activationFn;
};

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn);
seqmodel_layer_forward dense_layer_forward;
seqmodel_layer_backward dense_layer_backward;
seqmodel_layer_update_weights dense_layer_update_weights;

struct inputlayer_props
{
    int numInputs;
};

struct seqmodel_layer *input_layer_init(int numInputs);
seqmodel_layer_forward input_layer_forward;
seqmodel_layer_backward input_layer_backward;
seqmodel_layer_update_weights input_layer_update_weights;
