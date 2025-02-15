#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn)
{
    struct denselayer_props *props = mm_alloc(sizeof(struct denselayer_props));
    props->numNodes = numNodes;
    props->numInputs = numInputs;

    int weightsShape[2] = {numNodes, numInputs};
    props->weights = t_alloc(2, weightsShape);
    t_init_xavier(props->weights);

    int biasShape[1] = {numNodes};
    props->bias = t_alloc(1, biasShape);
    t_init_xavier(props->bias);

    props->activationFn = activationFn;

    return seqmodel_layer_init(
        props,
        "DENSE",
        dense_layer_forward,
        dense_layer_backward,
        dense_layer_update,
        dense_layer_mark_props);
}

void dense_layer_mark_props(void *props)
{
    struct denselayer_props *dp = (struct denselayer_props *)props;
    t_mark(dp->bias);
    t_mark(dp->weights);
}

tensor *dense_layer_forward(void *p, tensor *inputs, struct forwardstate *state)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;

    // activations = activationFn(weights * input + bias)
    tensor *dotProduct = t_mul(dp->weights, inputs);
    tensor *preActivations = t_elem_add(t_copy(dotProduct), dp->bias);
    tensor *activations = dp->activationFn(t_copy(preActivations), FUNCS_NORMAL);

    if (state != NULL)
    {
        state->preActivations = preActivations;
        state->activations = activations;
    }
    else
    {
        t_free(dotProduct);
        t_free(preActivations);
    }

    return activations;
}

void dense_layer_update(void *p, tensor *updateWeights, tensor *updateBias)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;

    // weights = weights - (weightGradients * updateFactor)
    t_elem_sub(dp->weights, updateWeights);

    // bias = bias - (biasGradients * updateFactor)
    t_elem_sub(dp->bias, updateBias);
}

struct backwardstate *dense_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;
    struct backwardstate *bs = backwardstate_alloc();

    tensor *activationDerivative = dp->activationFn(t_copy(curr->preActivations), FUNCS_DERIVATIVE);
    tensor *smallDelta = t_elem_mul(t_copy(previousSmallDelta), activationDerivative);

    // calculate nextSmallDelta
    tensor *weights_t = t_transpose(dp->weights, 2);
    tensor *nextSmallDelta = t_mul(weights_t, smallDelta);

    tensor *activations_t = t_transpose(prev->activations, 1);

    tensor *deltaW = t_mul(smallDelta, activations_t);

    bs->smallDelta = nextSmallDelta;
    bs->weightGradients = deltaW;
    bs->biasGradients = smallDelta;

    t_free(weights_t);
    t_free(activations_t);
    t_free(activationDerivative);

    return bs;
}