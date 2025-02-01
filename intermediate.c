#include <stdio.h>
#include <stdlib.h>

#include "mat.h"
#include "intermediate.h"

struct intermediate *intermediate_alloc(int nOutputs)
{
    struct intermediate *i = malloc(sizeof(struct intermediate));
    i->nOutputs = nOutputs;
    return i;
}

void intermediate_free(struct intermediate *i)
{
    vec_free(i->activations);
    vec_free(i->preActivations);
    free(i);
}