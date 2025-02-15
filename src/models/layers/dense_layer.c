#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *dense_layer_init(int numNodes, int numInputs, activationfunc *activationFn)
{
    int weightsShape[2] = {numNodes, numInputs};
    tensor *weights = t_alloc(2, weightsShape);
    t_init_xavier(weights);

    int biasShape[1] = {numNodes};
    tensor *bias = t_alloc(1, biasShape);
    t_init_xavier(bias);

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
    // activations = activationFn(weights * input + bias)
    tensor *dotProduct = t_mul(self->weights, inputs);
    tensor *preActivations = t_elem_add(t_copy(dotProduct), self->bias);
    tensor *activations = self->activationFn(t_copy(preActivations), FUNCS_NORMAL);

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

struct backwardstate *dense_layer_backward(struct seqmodel_layer *self, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    struct backwardstate *bs = backwardstate_alloc();

    tensor *activationDerivative = self->activationFn(t_copy(curr->preActivations), FUNCS_DERIVATIVE);
    tensor *smallDelta = t_elem_mul(t_copy(previousSmallDelta), activationDerivative);

    // calculate nextSmallDelta
    tensor *weights_t = t_transpose(self->weights, 2);
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