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

void forwardstate_lock(struct forwardstate *f)
{
    t_lock(f->activations);
    t_lock(f->preActivations);
}

struct backwardstate *backwardstate_alloc(int numNodes, int numInputs)
{
    struct backwardstate *bs = malloc(sizeof(struct backwardstate));
    bs->numNodes = numNodes;
    bs->numInputs = numInputs;
    return bs;
}

struct backwardstate *backwardstate_copy(struct backwardstate *src)
{
    if (src == NULL)
        return NULL;
    struct backwardstate *dst = backwardstate_alloc(src->numNodes, src->numInputs);
    dst->biasGradients = t_copy(src->biasGradients);
    dst->weightGradients = t_copy(src->weightGradients);
    return dst;
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
        tensor *factoredWeightGradients = t_mul_const(t_copy(src->weightGradients), factor);
        tensor *factoredBiasGradients = t_mul_const(t_copy(src->biasGradients), factor);
        t_elem_add(dst->weightGradients, factoredWeightGradients);
        t_elem_add(dst->biasGradients, factoredBiasGradients);
        t_free(factoredWeightGradients);
        t_free(factoredBiasGradients);
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

void backwardstate_lock(struct backwardstate *b)
{
    if (b == NULL)
        return;
    t_lock(b->biasGradients);
    t_lock(b->weightGradients);
    t_lock(b->smallDelta);
}