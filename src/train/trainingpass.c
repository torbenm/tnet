#include <stdlib.h>

#include "core.h"
#include "train.h"

struct trainingpass *trainingpass_init(
    param_t loss,
    struct backwardstate *backwardstates,
    int numLayers)
{
    struct trainingpass *tp = malloc(sizeof(struct trainingpass));
    tp->loss = loss;
    tp->numLayers = numLayers;
    tp->backwardstates = backwardstates;
    return tp;
}

void trainingpass_free(struct trainingpass *tp)
{
    for (int l = 0; l < tp->numLayers; l++)
    {
        // if (&tp->backwardstates[l] != NULL)
        // backwardstate_free(&tp->backwardstates[l]);
    }
    free(tp);
}

void trainingpass_lock(struct trainingpass *tp)
{
    for (int l = 0; l < tp->numLayers; l++)
    {
        if (&tp->backwardstates[l] != NULL)
            backwardstate_lock(&tp->backwardstates[l]);
    }
}