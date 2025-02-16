#include <stdlib.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

struct seqmodel_layer *input_layer_init()
{
    return seqmodel_layer_init("INPUT", NULL, NULL, NULL, input_layer_forward, NULL);
}

tensor *input_layer_forward(struct seqmodel_layer *self, tensor *inputs, struct forwardstate *state)
{
    tensor *outInputs = inputs;
    if (inputs->ndim == 1)
    {
        outInputs = t_prepend_dim(inputs);
    }
    if (state != NULL)
    {
        state->preActivations = t_copy(outInputs);
        state->activations = t_copy(outInputs);
    }
    return outInputs;
}