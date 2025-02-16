#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *dense_layer_init(int numInputs, int numNodes, activationfunc *activationFn)
{
    int weightsShape[2] = {numInputs, numNodes};
    tensor *weights = t_alloc(2, weightsShape);
    t_init_xavier(weights);

    int biasShape[2] = {numNodes};
    tensor *bias = t_alloc(1, biasShape);
    // t_init_xavier(bias);

    return seqmodel_layer_init(
        "DENSE",
        weights,
        bias,
        activationFn,
        dense_layer_forward,
        dense_layer_backward);
}

tensor *dense_layer_forward(struct seqmodel_layer *self, tensor *inputs, struct forwardstate *state)
{
    /**
     * Dense layer forward pass.
     * Math:
     *
     * out = activation(inputs x weights + bias)
     *
     *
     * Shapes:
     * inputs = (#batchsize, #inputs)
     * weights = (#inputs, #nodes)
     * bias = (#nodes)
     * out = (#nodes)
     */
    // (1, #inputs) x (#inputs, #nodes) = (#batchsize, #nodes)
    tensor *dotProduct = t_mul(inputs, self->weights);

    // hotfix for if dotProduct has two dims (standardcase)
    tensor *applyBias = self->bias;
    if (dotProduct->ndim > 1)
        applyBias = t_prepend_dim(self->bias);

    // (#batchsize, #nodes) + (#nodes) = (#batchsize, #nodes)
    tensor *preActivations = t_elem_add(t_copy(dotProduct), applyBias);
    // act((#batchsize, #nodes)) = (#batchsize, #nodes)
    tensor *activations = self->activationFn(t_copy(preActivations), FUNCS_NORMAL);

    if (state != NULL)
    {
        state->preActivations = preActivations;
        state->activations = activations;
    }
    else
    {
        if (dotProduct->ndim > 1)
            t_free(applyBias);
        t_free(dotProduct);
        t_free(preActivations);
    }

    return activations;
}

struct backwardstate *dense_layer_backward(struct seqmodel_layer *self, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    /**
     * Dense layer backward pass. We calculate the gradients here.
     *
     * Math:
     * smallDelta = previousSmallDelta * activation'(curr->prevActivation)
     * weightGradients = smallDelta x (prev->activations^T)
     * biasGradients = smallDelta
     * nextSmallDelta = weights^T x smallDelta
     *
     * Shapes:
     * previousSmallDelta = (#batchsize, #nodes)
     * curr->prevActivations = (#batchsize, #nodes)
     * prev->activations = (#batchsize, #inputs)
     * weights = (#inputs, #nodes)
     * bias = (#nodes)
     */
    // act'((#batchsize, #nodes)) = (#batchsize, #nodes)
    tensor *activationDerivative = self->activationFn(t_copy(curr->preActivations), FUNCS_DERIVATIVE);
    // (#batchsize, #nodes) * (#batchsize, #nodes) = (#batchsize, #nodes)
    tensor *smallDelta = t_elem_mul(t_copy(previousSmallDelta), activationDerivative);
    // (#batchsize, #inputs)^T = (#inputs, #batchsize)
    tensor *activations_t = t_transpose(prev->activations, 2);
    // (#inputs, #batchsize) x (#batchsize, #nodes) = (#inputs, #nodes)
    tensor *deltaW = t_mul(activations_t, smallDelta);
    // (#inputs, #nodes)^T = (#nodes, #inputs)
    tensor *weights_t = t_transpose(self->weights, 2);
    // (#batchsize, #nodes) x (#nodes, #inputs) = (#batchsize, #inputs)
    tensor *nextSmallDelta = t_mul(smallDelta, weights_t);
    // ensure we have two dimensions with deltaW (in case #batchsize = 1)
    if (deltaW->ndim == 1)
        deltaW = t_append_dim(deltaW);

    struct backwardstate *bs = backwardstate_alloc();
    bs->smallDelta = nextSmallDelta;
    bs->weightGradients = deltaW;
    bs->biasGradients = smallDelta;

    t_free(weights_t);
    t_free(activations_t);
    t_free(activationDerivative);

    return bs;
}