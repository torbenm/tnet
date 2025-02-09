#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn)
{
    struct denselayer_props *props = malloc(sizeof(struct denselayer_props));
    props->numNodes = numNodes;
    props->numInputs = numInputs;

    int weightsShape[2] = {numNodes, numInputs};
    props->weights = t_alloc(2, weightsShape);
    t_init_rand(props->weights);

    int biasShape[1] = {numNodes};
    props->bias = t_alloc(1, biasShape);
    t_init_rand(props->bias);

    props->activationFn = activationFn;

    // move somewhere else...
    struct seqmodel_layer *l = malloc(sizeof(struct seqmodel_layer));
    l->layerProps = props;
    l->numOutputs = numNodes;
    l->forward = dense_layer_forward;
    l->backward = dense_layer_backward;
    l->update = dense_layer_update_weights;
    return l;
}

tensor *dense_layer_forward(void *p, tensor *inputs, struct forwardstate *state)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;

    // activations = activationFn(weights * input + bias)
    tensor *dotProduct = t_mul(t_copy(dp->weights), inputs);
    tensor *preActivations = t_elem_add(t_copy(dotProduct), dp->bias);
    tensor *activations = dp->activationFn(t_copy(preActivations), FUNCS_NORMAL);

    if (state != NULL)
    {
        state->nOutputs = dp->numNodes;
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

void dense_layer_update_weights(void *p, struct backwardstate *bs, param_t updateFactor)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;
    // weights = weights - (weightGradients * updateFactor)
    tensor *factored_weightGradients = t_mul_const(t_copy(bs->weightGradients), updateFactor);
    t_elem_sub(dp->weights, factored_weightGradients);
    t_free(factored_weightGradients);

    // bias = bias - (biasGradients * updateFactor)
    tensor *factored_biasGradients = t_mul_const(t_copy(bs->biasGradients), updateFactor);
    t_elem_sub(dp->bias, factored_biasGradients);
    t_free(factored_biasGradients);
}

struct backwardstate *dense_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev, param_t learningRate)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;
    struct backwardstate *bs = backwardstate_alloc(dp->numNodes, dp->numInputs);

    tensor *activationDerivative = dp->activationFn(t_copy(curr->preActivations), FUNCS_DERIVATIVE);
    tensor *smallDelta = t_elem_mul(t_copy(previousSmallDelta), activationDerivative);

    // calculate nextSmallDelta
    tensor *weights_t = t_transpose(dp->weights, 2);
    tensor *nextSmallDelta = t_mul(weights_t, smallDelta);

    // TODO: this still needs to be implemented properly and thought through
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