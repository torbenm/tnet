#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"
#include "train.h"
#include "funcs.h"

struct optimizer *opt_sgd_init(param_t learningRate, param_t monumentum, lossfunc *lossFn)
{
    struct optimizer *o = malloc(sizeof(struct optimizer));
    o->numParams = 2;
    o->params = malloc(o->numParams * sizeof(param_t));
    o->params[0] = learningRate;
    o->params[1] = monumentum;
    o->run_opt = opt_sgd;
    o->lossFn = lossFn;
    return o;
}

struct trainingpass *opt_sgd(struct seqmodel *seq, param_t *params, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn, struct trainingpass *previouspass)
{

    struct forwardstate **forwardstates = malloc(batchSize * sizeof(struct forwardstate *));
    tensor *predictions[batchSize];

    param_t learningRate = params[0];
    param_t monumentum = params[1];

    /**
     * Forwardpass
     */
    for (int t = 0; t < batchSize; t++)
    {
        forwardstates[t] = malloc((seq->numLayers) * sizeof(struct forwardstate));
        for (int l = 0; l < seq->numLayers; l++)
        {
            if (l == 0)
                seq->layers[l]->forward(seq->layers[l]->layerProps, inputs[t], &forwardstates[t][l]);
            else
                seq->layers[l]->forward(seq->layers[l]->layerProps, forwardstates[t][l - 1].activations, &forwardstates[t][l]);
            forwardstate_lock(&forwardstates[t][l]);
        }
        predictions[t] = forwardstates[t][seq->numLayers - 1].activations;
    }

    /**
     * Backwardpass
     */

    struct backwardstate *globalbackwardstates = malloc(seq->numLayers * sizeof(struct backwardstate));

    for (int t = 0; t < batchSize; t++)
    {
        // initialize with derivative of mse (TODO: replace with function call)
        tensor *nextDelta = t_elem_sub(t_copy(predictions[t]), truths[t]);
        struct backwardstate **localbackwardstates = malloc(seq->numLayers * sizeof(struct backwardstate *));
        // Calculate deltas pass
        for (int l = seq->numLayers - 1; l >= 0; l--)
        {
            struct forwardstate *prev = NULL;
            if (l > 0)
                prev = &forwardstates[t][l - 1];
            localbackwardstates[l] = seq->layers[l]->backward(
                seq->layers[l]->layerProps,
                nextDelta,
                &forwardstates[t][l],
                prev,
                learningRate);

            if (localbackwardstates[l] != NULL)
            {
                t_free(nextDelta);
                nextDelta = localbackwardstates[l]->smallDelta;
            }
        }
        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {

            struct backwardstate *weightedBs = backwardstate_copy(localbackwardstates[l]);

            if (monumentum > 0 && previouspass != NULL && weightedBs != NULL && &previouspass->backwardstates[l] != NULL)
            {
                // previous update weights * monumentum
                tensor *mon_w_prev = t_mul_const(t_copy(previouspass->backwardstates[l].weightGradients), monumentum);
                tensor *mon_b_prev = t_mul_const(t_copy(previouspass->backwardstates[l].biasGradients), monumentum);

                t_mul_const(weightedBs->weightGradients, 1 - monumentum);
                t_mul_const(weightedBs->biasGradients, 1 - monumentum);

                t_elem_add(weightedBs->weightGradients, mon_w_prev);
                t_elem_add(weightedBs->biasGradients, mon_b_prev);

                t_free(mon_w_prev);
                t_free(mon_b_prev);
            }
            backwardstate_lock(weightedBs); // avoid its properties being altered.

            param_t batchSizeFactor = 1.0 / (param_t)batchSize;

            seq->layers[l]->update(
                seq->layers[l]->layerProps,
                weightedBs,
                batchSizeFactor * learningRate);

            // // updating the global backwardstate
            backwardstate_incorporate(&globalbackwardstates[l], localbackwardstates[l], batchSizeFactor);

            // // free memory - not needed any longer
            backwardstate_free(weightedBs);
            forwardstate_free(&forwardstates[t][l]);
        }
        free(localbackwardstates);
        free(forwardstates[t]);
    }

    /**
     * Prediction pass
     */
    for (int t = 0; t < batchSize; t++)
    {
        predictions[t] = seqmodel_predict(seq, inputs[t]);
    }
    return trainingpass_init(lossFn(batchSize, predictions, truths), globalbackwardstates, seq->numLayers);
}