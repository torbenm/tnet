#include <stdlib.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *input_layer_init(int numInputs)
{
    struct inputlayer_props *props = mm_alloc(sizeof(struct inputlayer_props));
    props->numInputs = numInputs;

    // move somewhere else...
    struct seqmodel_layer *l = mm_alloc(sizeof(struct seqmodel_layer));
    l->layerProps = props;
    l->numOutputs = numInputs;
    l->forward = input_layer_forward;
    l->backward = input_layer_backward;
    l->update = input_layer_update_weights;
    l->mark = input_layer_mark;
    return l;
}

void input_layer_mark(struct seqmodel_layer *self)
{
    mm_mark(self->layerProps);
    mm_mark(self);
}

tensor *input_layer_forward(void *p, tensor *inputs, struct forwardstate *state)
{
    struct inputlayer_props *ip = (struct inputlayer_props *)p;

    if (state != NULL)
    {
        state->nOutputs = ip->numInputs;
        state->preActivations = t_copy(inputs);
        state->activations = t_copy(inputs);
    }

    return inputs;
}

struct backwardstate *input_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev, param_t learningRate)
{
    return NULL; // should _not_ be used!
}

void input_layer_update_weights(void *p, tensor *updateWeights, tensor *updateBias)
{
    // NOOP
}