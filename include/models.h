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
typedef struct seqmodel_layer seqmodel_layer;

typedef tensor *seqmodel_layer_forward(seqmodel_layer *self, tensor *input, struct forwardstate *state);
typedef struct backwardstate *seqmodel_layer_backward(seqmodel_layer *self, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev);
typedef void seqmodel_layer_update(seqmodel_layer *self, tensor *updateWeights, tensor *updateBias);

struct seqmodel_layer
{
    const char *name;
    tensor *weights;
    tensor *bias;
    activationfunc *activationFn;
    seqmodel_layer_forward *forward;
    seqmodel_layer_backward *backward;
    seqmodel_layer_update *update;
};

struct seqmodel_layer *seqmodel_layer_init(const char *name, tensor *weights, tensor *bias, activationfunc *activationFn, seqmodel_layer_forward *forward, seqmodel_layer_backward *backward);
void seqmodel_layer_mark(struct seqmodel_layer *self);
void seqmodel_layer_print(struct seqmodel_layer *self);

typedef struct seqmodel
{
    int numLayers;
    int _layerBufferSize;
    struct seqmodel_layer **layers;
} seqmodel;
struct seqmodel *seqmodel_init();
void seqmodel_resize(struct seqmodel *seq, int newSize);
void seqmodel_push(struct seqmodel *seq, struct seqmodel_layer *layer);
void seqmodel_mark(struct seqmodel *seq);
void seqmodel_print(struct seqmodel *s);
tensor *seqmodel_predict(struct seqmodel *seq, tensor *input);
param_t seqmodel_calculate_loss(struct seqmodel *seq, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], loss *loss);

/**
 * Fully Connected / Dense Layer
 */

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn);
seqmodel_layer_forward dense_layer_forward;
seqmodel_layer_backward dense_layer_backward;

/**
 * Input Layer - may be used as initial layer.
 */
struct seqmodel_layer *input_layer_init();
seqmodel_layer_forward input_layer_forward;

/**
 * Activations - Softmax Layer
 */
struct seqmodel_layer *softmax_layer_init();
seqmodel_layer_forward softmax_layer_forward;
seqmodel_layer_backward softmax_layer_backward;