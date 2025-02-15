#pragma once

#include "core.h"
#include "funcs.h"

struct forwardstate
{
    tensor *activations;
    tensor *preActivations;
};

struct forwardstate *forwardstate_alloc();
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

struct backwardstate *backwardstate_alloc();
void backwardstate_free(struct backwardstate *);
void backwardstate_lock(struct backwardstate *);
void backwardstate_mark(struct backwardstate *);

/**
 * Seqmodel & its possible models
 */
typedef tensor *seqmodel_layer_forward(void *layer_struct, tensor *input, struct forwardstate *state);
typedef struct backwardstate *seqmodel_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev);
typedef void seqmodel_layer_update_weights(void *p, tensor *updateWeights, tensor *updateBias);
struct seqmodel_layer
{
    void *layerProps;
    const char *name;
    seqmodel_layer_forward *forward;
    seqmodel_layer_backward *backward;
    seqmodel_layer_update_weights *update;
    void (*markProps)(void *);
};
struct seqmodel_layer *seqmodel_layer_init(void *props, const char *name, seqmodel_layer_forward *forward, seqmodel_layer_backward *backward, seqmodel_layer_update_weights *update, void (*markProps)(void *));
void seqmodel_layer_mark(struct seqmodel_layer *self);
void seqmodel_layer_print(struct seqmodel_layer *self);

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
void seqmodel_print(struct seqmodel *s);
tensor *seqmodel_predict(struct seqmodel *seq, tensor *input);
param_t seqmodel_calculate_loss(struct seqmodel *seq, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn);

/**
 * Fully Connected / Dense Layer
 */
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
seqmodel_layer_update_weights dense_layer_update;
void dense_layer_mark_props(void *props);

struct inputlayer_props
{
    int numInputs;
};

/**
 * Input Layer - may be used as initial layer.
 */
struct seqmodel_layer *input_layer_init(int numInputs);
seqmodel_layer_forward input_layer_forward;

/**
 * Activations - Softmax Layer
 */
struct seqmodel_layer *softmax_layer_init();
seqmodel_layer_forward softmax_layer_forward;
seqmodel_layer_backward softmax_layer_backward;