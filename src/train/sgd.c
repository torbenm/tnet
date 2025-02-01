#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"
#include "train.h"
#include "funcs.h"

struct optimizer *opt_sgd_init(param_t learningRate, lossfunc *lossFn)
{
    struct optimizer *o = malloc(sizeof(struct optimizer));
    o->numParams = 1;
    o->params = malloc(o->numParams * sizeof(param_t));
    o->params[0] = learningRate;
    o->run_opt = opt_sgd;
    o->lossFn = lossFn;
    return o;
}

param_t opt_sgd(struct seqmodel *seq, param_t *params, int numExamples, vec inputs[numExamples], vec truths[numExamples], lossfunc *lossFn)
{
    setbuf(stdout, 0);

    struct forwardstate **forwardstates = malloc(numExamples * sizeof(struct forwardstate *));
    vec predictions[numExamples];

    param_t learningRate = params[0];

    /**
     * Forwardpass
     */
    for (int t = 0; t < numExamples; t++)
    {
        forwardstates[t] = malloc((seq->numLayers) * sizeof(struct forwardstate));
        for (int l = 0; l < seq->numLayers; l++)
        {
            if (l == 0)
                seq->layers[l]->forward(seq->layers[l]->layerProps, inputs[t], &forwardstates[t][l]);
            else
                seq->layers[l]->forward(seq->layers[l]->layerProps, forwardstates[t][l - 1].activations, &forwardstates[t][l]);
        }
        predictions[t] = forwardstates[t][seq->numLayers - 1].activations;
    }

    /**
     * Backwardpass
     */
    for (int t = 0; t < numExamples; t++)
    {
        vec nextDelta = vec_elem_sub(predictions[t], truths[t], seq->layers[seq->numLayers - 1]->numOutputs);

        struct backwardstate **backwardstates = malloc(seq->numLayers * sizeof(struct backwardstate *));

        // Calculate deltas pass
        for (int l = seq->numLayers - 1; l >= 0; l--)
        {
            struct forwardstate *prev = NULL;
            if (l > 0)
                prev = &forwardstates[t][l - 1];
            backwardstates[l] = seq->layers[l]->backward(
                seq->layers[l]->layerProps,
                nextDelta,
                &forwardstates[t][l],
                prev,
                learningRate,
                (l == (seq->numLayers - 1)));

            if (backwardstates[l] != NULL)
                nextDelta = backwardstates[l]->smallDelta;
        }

        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {
            seq->layers[l]->update(
                seq->layers[l]->layerProps,
                backwardstates[l],
                (1.0 / (param_t)numExamples) * learningRate);
            // free memory - not needed any longer
            backwardstate_free(backwardstates[l]);
            forwardstate_free(&forwardstates[t][l]);
        }
        free(backwardstates);
        free(forwardstates[t]);
    }

    /**
     * Prediction pass
     */
    for (int t = 0; t < numExamples; t++)
    {
        predictions[t] = seqmodel_predict(seq, inputs[t]);
    }
    return lossFn(numExamples, 1, predictions, truths);
}