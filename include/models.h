#pragma once

#include "core.h"
#include "funcs.h"

struct forwardstate
{
    tensor *activations;
    tensor *preActivations;
    int nOutputs;
};

struct forwardstate *forwardstate_alloc(int size);
void forwardstate_free(struct forwardstate *);
void forwardstate_mark(struct forwardstate *);
void forwardstate_lock(struct forwardstate *f);

struct backwardstate
{
    int numNodes;
    int numInputs;
    tensor *weightGradients;
    tensor *biasGradients;
    tensor *smallDelta;
};

struct backwardstate *backwardstate_alloc(int numNodes, int numInputs);
void backwardstate_free(struct backwardstate *);
void backwardstate_lock(struct backwardstate *);
void backwardstate_mark(struct backwardstate *);

// Perceptron - tensor
struct perceptron
{
    tensor *bias;
    tensor *weights;
    activationfunc *activationFn;
};

tensor *perceptron_forward(struct perceptron *p, tensor *vals);
int perceptron_train(struct perceptron *p, int maxIter, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], param_t learningRate);
struct perceptron *perceptron_init(int numWeights);
void perceptron_free(struct perceptron *p);

/**
 * NEW Seqmodel & its possible models
 */
#define SEQMODEL_STD_SIZE 4

typedef tensor *seqmodel_layer_forward(void *layer_struct, tensor *input, struct forwardstate *state);
typedef struct backwardstate *seqmodel_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev);
typedef void seqmodel_layer_update_weights(void *p, tensor *updateWeights, tensor *updateBias);
struct seqmodel_layer
{
    void *layerProps;
    int numOutputs;
    seqmodel_layer_forward *forward;
    seqmodel_layer_backward *backward;
    seqmodel_layer_update_weights *update;
    void (*mark)(struct seqmodel_layer *);
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
void seqmodel_mark(struct seqmodel *seq);
tensor *seqmodel_predict(struct seqmodel *seq, tensor *input);
param_t seqmodel_calculate_loss(struct seqmodel *seq, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn);

struct denselayer_props
{
    int numNodes;
    int numInputs;
    tensor *weights;
    tensor *bias;
    activationfunc *activationFn;
};

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn);
seqmodel_layer_forward dense_layer_forward;
seqmodel_layer_backward dense_layer_backward;
seqmodel_layer_update_weights dense_layer_update_weights;
void dense_layer_mark(struct seqmodel_layer *self);

struct inputlayer_props
{
    int numInputs;
};

struct seqmodel_layer *input_layer_init(int numInputs);
seqmodel_layer_forward input_layer_forward;
seqmodel_layer_backward input_layer_backward;
seqmodel_layer_update_weights input_layer_update_weights;
void input_layer_mark(struct seqmodel_layer *self);
