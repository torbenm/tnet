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
        dst->weightGradients = mat_mul_const(src->weightGradients, factor, src->numNodes, src->numInputs);
        dst->biasGradients = vec_mul_const(src->biasGradients, factor, src->numNodes);
    }
    else
    {
        mat newWeightGradients = mat_elem_add_mul(
            dst->weightGradients,
            src->weightGradients,
            factor,
            src->numNodes,
            src->numInputs);
        vec newBiasGradients = vec_elem_add_mul(
            dst->biasGradients,
            src->biasGradients,
            factor,
            src->numNodes);
        mat_free(dst->weightGradients, dst->numNodes);
        vec_free(dst->biasGradients);
        dst->weightGradients = newWeightGradients;
        dst->biasGradients = newBiasGradients;
    }
}

void backwardstate_free(struct backwardstate *i)
{
    if (i == NULL)
        return;
    vec_free(i->biasGradients);
    mat_free(i->weightGradients, i->numNodes);
    vec_free(i->smallDelta);
    free(i);
}