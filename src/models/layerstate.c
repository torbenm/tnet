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
    vec_free(i->activations);
    vec_free(i->preActivations);
    free(i);
}

struct backwardstate *backwardstate_alloc()
{
    return malloc(sizeof(struct backwardstate));
}

void backwardstate_free(struct backwardstate *i)
{
    vec_free(i->bias_gradients);
    // TODO mat_free(i->weight_gradients);
    vec_free(i->delta);
    free(i);
}