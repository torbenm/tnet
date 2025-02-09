#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"

struct forwardstate *forwardstate_alloc(int nOutputs)
{
    struct forwardstate *i = malloc(sizeof(struct forwardstate));
    i->nOutputs = nOutputs;
    return i;
}

void forwardstate_free(struct forwardstate *i)
{
    // vec_free(i->activations);
    // vec_free(i->preActivations);
    // free(i);
}

struct backwardstate *backwardstate_alloc(int numNodes, int numInputs)
{
    struct backwardstate *bs = malloc(sizeof(struct backwardstate));
    bs->numNodes = numNodes;
    bs->numInputs = numInputs;
    return bs;
}

void backwardstate_incorporate(struct backwardstate *dst, struct backwardstate *src, param_t factor)
{
    if (src == NULL)
        return;
    if (dst->weightGradients == NULL || dst->biasGradients == NULL)
    {
        dst->numInputs = src->numInputs;
        dst->numNodes = src->numNodes;
        dst->weightGradients = t_mul_const(t_copy(src->weightGradients), factor);
        dst->biasGradients = t_mul_const(t_copy(src->biasGradients), factor);
    }
    else
    {
        t_elem_add(dst->weightGradients, t_mul_const(src->weightGradients, factor));
        t_elem_add(dst->biasGradients, t_mul_const(src->biasGradients, factor));
    }
}

void backwardstate_free(struct backwardstate *i)
{
    if (i == NULL)
        return;
    t_free(i->biasGradients);
    t_free(i->weightGradients);
    t_free(i->smallDelta);
    free(i);
}