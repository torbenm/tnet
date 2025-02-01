#include <stdlib.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *input_layer_init(int numInputs)
{
    struct inputlayer_props *props = malloc(sizeof(struct inputlayer_props));
    props->numInputs = numInputs;

    // move somewhere else...
    struct seqmodel_layer *l = malloc(sizeof(struct seqmodel_layer));
    l->layerProps = props;
    l->numOutputs = numInputs;
    l->forward = input_layer_forward;
    l->backward = input_layer_backward;
    return l;
}

vec input_layer_forward(void *p, vec inputs, struct forwardstate *state)
{
    struct inputlayer_props *ip = (struct inputlayer_props *)p;

    if (state != NULL)
    {
        state->nOutputs = ip->numInputs;
        state->preActivations = inputs;
        state->activations = inputs;
    }

    return inputs;
}

vec input_layer_backward(void *p, vec previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev, param_t learningRate, int isOutputLayer)
{
    return NULL; // should _not_ be used!
}