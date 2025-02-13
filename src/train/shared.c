#include <stdio.h>

#include "core.h"
#include "models.h"
#include "train.h"

struct forwardstate *opt_forwardpropagate(struct seqmodel *seq, tensor *inputs, tensor **outPredictions)
{

    struct forwardstate *forwardstates = mm_alloc((seq->numLayers) * sizeof(struct forwardstate));
    for (int l = 0; l < seq->numLayers; l++)
    {
        if (l == 0)
            seq->layers[l]->forward(seq->layers[l]->layerProps, inputs, &forwardstates[l]);
        else
            seq->layers[l]->forward(seq->layers[l]->layerProps, forwardstates[l - 1].activations, &forwardstates[l]);
        forwardstate_lock(&forwardstates[l]);
    }
    *(outPredictions) = forwardstates[seq->numLayers - 1].activations;
    return forwardstates;
}

struct backwardstate **opt_backwardpropagate(struct seqmodel *seq, tensor *prediction, tensor *truth, struct forwardstate *forwardstates)
{
    // initialize with derivative of mse (TODO: replace with function call)
    tensor *nextDelta = t_elem_sub(t_copy(prediction), truth);

    struct backwardstate **backwardstates = mm_alloc(seq->numLayers * sizeof(struct backwardstate *));
    // Calculate deltas pass
    for (int l = seq->numLayers - 1; l >= 0; l--)
    {
        struct forwardstate *prev = NULL;
        if (l > 0)
            prev = &forwardstates[l - 1];
        backwardstates[l] = seq->layers[l]->backward(
            seq->layers[l]->layerProps,
            nextDelta,
            &forwardstates[l],
            prev);

        t_free(nextDelta);
        if (l < seq->numLayers - 1)
            backwardstates[l + 1]->smallDelta = NULL;

        if (backwardstates[l] != NULL)
            nextDelta = backwardstates[l]->smallDelta;
    }
    t_free(nextDelta);

    return backwardstates;
}
