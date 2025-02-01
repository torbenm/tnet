#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"

struct layerstate *layerstate_alloc(int nOutputs)
{
    struct layerstate *i = malloc(sizeof(struct layerstate));
    i->nOutputs = nOutputs;
    return i;
}

void layerstate_free(struct layerstate *i)
{
    vec_free(i->activations);
    vec_free(i->preActivations);
    free(i);
}