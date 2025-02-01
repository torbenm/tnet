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

struct backwardstate *backwardstate_alloc(int numNodes)
{
    struct backwardstate *bs = malloc(sizeof(struct backwardstate));
    bs->numNodes = numNodes;
    return bs;
}

void backwardstate_free(struct backwardstate *i)
{
    if (i == NULL)
        return;
    vec_free(i->biasDelta);
    mat_free(i->weightDelta, i->numNodes);
    vec_free(i->smallDelta);
    free(i);
}