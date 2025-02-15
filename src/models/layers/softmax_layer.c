#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

/**
 * Actual softmax implementation.

 */
tensor *softmax(tensor *t)
{
    param_t abs_max = 0;

    for (int i = 0; i < t->_v_size; i++)
        abs_max = fmax(fabs(t->v[i]), abs_max);

    param_t sumOfAll = 0;
    for (int i = 0; i < t->_v_size; i++)
        sumOfAll += exp(t->v[i] - abs_max);

    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = exp(t->v[i] - abs_max) / (sumOfAll);
    }
    return t;
}

struct seqmodel_layer *softmax_layer_init()
{
    return seqmodel_layer_init(
        NULL,
        "SOFTMAX",
        softmax_layer_forward,
        softmax_layer_backward,
        NULL,
        NULL);
}

tensor *softmax_layer_forward(void *p, tensor *inputs, struct forwardstate *state)
{
    tensor *t = softmax(t_copy(inputs));

    if (state != NULL)
    {
        state->activations = t_copy(t);
        state->preActivations = t_copy(inputs);
    }

    return t;
}

struct backwardstate *softmax_layer_backward(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    struct backwardstate *bs = backwardstate_alloc();

    tensor *s_t = t_transpose(curr->activations, 1);
    tensor *s_diag = t_diag(curr->activations);
    tensor *s_mul = t_mul(curr->activations, s_t);

    t_elem_sub(s_diag, s_mul); // stored in s_diag

    tensor *smallDeltaT = t_transpose(previousSmallDelta, 1);

    bs->smallDelta = t_flatten(t_mul(smallDeltaT, s_diag));
    bs->biasGradients = NULL;   // shouldn't be needed
    bs->weightGradients = NULL; // shouldn't be needed

    t_free(s_t);
    t_free(s_mul);

    return bs;
}