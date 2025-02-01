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
    props->weights = mat_alloc_rand(numNodes, numInputs);
    props->bias = vec_alloc_rand(numNodes);
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

vec dense_layer_forward(void *p, vec inputs, struct forwardstate *state)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;

    vec dotProduct = mat_dot_product(dp->weights, inputs, dp->numNodes, dp->numInputs);
    vec preActivations = vec_elem_add(dotProduct, dp->bias, dp->numNodes);
    vec activations = dp->activationFn(preActivations, dp->numNodes, ACTIVATION_FORWARD);

    if (state != NULL)
    {
        state->nOutputs = dp->numNodes;
        state->preActivations = preActivations;
        state->activations = activations;
    }
    else
    {
        vec_free(dotProduct);
        vec_free(preActivations);
    }

    return activations;
}

void dense_layer_update_weights(void *p, struct backwardstate *bs, param_t updateFactor)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;
    mat newWeights = mat_elem_sub_mul(dp->weights, bs->weightGradients, updateFactor, dp->numNodes, dp->numInputs);
    vec newBias = vec_elem_sub_mul(dp->bias, bs->biasGradients, updateFactor, dp->numNodes);
    vec_free(dp->bias);
    mat_free(dp->weights, dp->numNodes);

    dp->weights = newWeights;
    dp->bias = newBias;
}

struct backwardstate *dense_layer_backward(void *p, vec previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev, param_t learningRate, int isOutputLayer)
{
    struct denselayer_props *dp = (struct denselayer_props *)p;
    struct backwardstate *bs = backwardstate_alloc(dp->numNodes, dp->numInputs);

    vec smallDelta = previousSmallDelta; // only one dimension for dense_layer
    if (!isOutputLayer)
    {
        // Middle layer - previousSmallDelta needs to be multiplied with activation derivation
        vec actDeriv = dp->activationFn(curr->preActivations, dp->numNodes, ACTIVATION_DERIVATIVE);
        smallDelta = vec_elem_mul(smallDelta, actDeriv, dp->numNodes);
        vec_free(actDeriv);
    }

    // calculate nextSmallDelta
    mat weights_t = mat_transpose(dp->weights, dp->numNodes, dp->numInputs);
    vec nextSmallDelta = mat_dot_product(weights_t, smallDelta, dp->numInputs, dp->numNodes);
    mat_free(weights_t, dp->numNodes);
    bs->smallDelta = nextSmallDelta;

    mat deltaW = vec_transposed_vec_mul(smallDelta, prev->activations, 1.0, dp->numNodes, dp->numInputs);
    bs->weightGradients = deltaW;
    bs->biasGradients = vec_mul_const(smallDelta, 1.0, dp->numNodes);

    return bs;
}