#include <stdlib.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *input_layer_init(int numInputs)
{
    return seqmodel_layer_init(NULL, "INPUT", input_layer_forward, NULL, NULL, NULL);
}

tensor *input_layer_forward(void *p, tensor *inputs, struct forwardstate *state)
{
    struct inputlayer_props *ip = (struct inputlayer_props *)p;

    if (state != NULL)
    {
        state->preActivations = t_copy(inputs);
        state->activations = t_copy(inputs);
    }

    return inputs;
}