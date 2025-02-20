#include <stdio.h>

#include "core.h"
#include "models.h"
#include "train.h"

void opt_fowardbackwardpass(struct seqmodel *s, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], tensor ***outWeightGradients, tensor ***outBiasGradients, loss *loss)
{
    tensor *predictions[batchSize];

    for (int t = 0; t < batchSize; t++)
    {
        struct forwardstate *forwardstates = opt_forwardpropagate(s, inputs[t], &predictions[t]);
        struct backwardstate **localbackwardstates = opt_backwardpropagate(s, predictions[t], truths[t], forwardstates, loss);
        // Apply deltas pass
        for (int l = 0; l < s->numLayers; l++)
        {
            // Only able to update weights if we have a backwardstate.
            // Some layers don't provide us one
            if (localbackwardstates[l] != NULL && localbackwardstates[l]->weightGradients != NULL && localbackwardstates[l]->biasGradients != NULL)
            {
                t_copy_or_add(&((*outWeightGradients)[l]), localbackwardstates[l]->weightGradients);
                t_copy_or_add(&((*outBiasGradients)[l]), localbackwardstates[l]->biasGradients);
            }
        }
    }

    for (int l = 0; l < s->numLayers; l++)
    {
        if ((*outWeightGradients)[l] != NULL && (*outBiasGradients)[l] != NULL)
        {
            t_div_const((*outWeightGradients)[l], (param_t)batchSize);
            t_div_const((*outBiasGradients)[l], (param_t)batchSize);
        }
    }
}

struct forwardstate *opt_forwardpropagate(struct seqmodel *seq, tensor *inputs, tensor **outPredictions)
{

    struct forwardstate *forwardstates = mm_alloc((seq->numLayers) * sizeof(struct forwardstate));
    for (int l = 0; l < seq->numLayers; l++)
    {
        if (l == 0)
            seq->layers[l]->forward(seq->layers[l], inputs, &forwardstates[l]);
        else
            seq->layers[l]->forward(seq->layers[l], forwardstates[l - 1].activations, &forwardstates[l]);
        forwardstate_lock(&forwardstates[l]);
    }
    *(outPredictions) = forwardstates[seq->numLayers - 1].activations;
    return forwardstates;
}

struct backwardstate **opt_backwardpropagate(struct seqmodel *seq, tensor *prediction, tensor *truth, struct forwardstate *forwardstates, loss *loss)
{
    tensor *initial_loss = loss->backward(prediction, truth);
    struct backwardstate **backwardstates = mm_alloc(seq->numLayers * sizeof(struct backwardstate *));
    // Calculate deltas pass
    tensor *nextDelta = initial_loss;
    for (int l = seq->numLayers - 1; l >= 0; l--)
    {
        struct forwardstate *prev = NULL;
        if (l > 0)
            prev = &forwardstates[l - 1];
        backwardstates[l] = seq->layers[l]->backward(
            seq->layers[l],
            nextDelta,
            &forwardstates[l],
            prev);

        if (l < seq->numLayers - 1)
            backwardstates[l + 1]->smallDelta = NULL;

        if (backwardstates[l] != NULL)
            nextDelta = backwardstates[l]->smallDelta;
    }
    t_free(initial_loss);

    return backwardstates;
}

void opt_mark(struct optimizer *o)
{
    mm_mark(o);
    mm_mark(o->loss);
    mm_mark(o->params);
}