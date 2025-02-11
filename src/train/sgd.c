#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"
#include "train.h"
#include "funcs.h"

struct optimizer *opt_sgd_init(param_t learningRate, param_t monumentum, lossfunc *lossFn)
{
    struct optimizer *o = mm_alloc(sizeof(struct optimizer));
    o->numParams = 2;
    o->params = mm_alloc(o->numParams * sizeof(param_t));
    o->params[0] = learningRate;
    o->params[1] = monumentum;
    o->run_opt = opt_sgd;
    o->lossFn = lossFn;
    return o;
}

void optimizer_mark(struct optimizer *o)
{
    mm_mark(o);
    mm_mark(o->params);
}

struct trainingpass *opt_sgd(struct seqmodel *seq, param_t *params, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn, struct trainingpass *previouspass, int trainingPassNum)
{

    struct forwardstate **forwardstates = mm_alloc(batchSize * sizeof(struct forwardstate *));
    tensor *predictions[batchSize];

    param_t learningRate = params[0];
    param_t monumentum = params[1];

    /**
     * Forwardpass
     */
    for (int t = 0; t < batchSize; t++)
    {
        forwardstates[t] = mm_alloc((seq->numLayers) * sizeof(struct forwardstate));
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

    struct tensor **stored_tensors = mm_calloc((seq->numLayers * 2), sizeof(struct tensor *));

    for (int t = 0; t < batchSize; t++)
    {
        // initialize with derivative of mse (TODO: replace with function call)
        tensor *nextDelta = t_elem_sub(t_copy(predictions[t]), truths[t]);

        struct backwardstate **localbackwardstates = mm_alloc(seq->numLayers * sizeof(struct backwardstate *));

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

            t_free(nextDelta);
            if (l < seq->numLayers - 1)
                localbackwardstates[l + 1]->smallDelta = NULL;

            if (localbackwardstates[l] != NULL)
                nextDelta = localbackwardstates[l]->smallDelta;
        }
        t_free(nextDelta);

        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {

            if (localbackwardstates[l] != NULL)
            {
                tensor *updateWeights = t_copy(localbackwardstates[l]->weightGradients);
                tensor *updateBias = t_copy(localbackwardstates[l]->biasGradients);

                param_t batchSizeFactor = 1.0 / (param_t)batchSize;

                int trainingpassWeightsIdx = l * 2;
                int trainingpassBiasIdx = l * 2 + 1;

                if (previouspass != NULL && previouspass->stored_tensors[trainingpassWeightsIdx] != NULL && previouspass->stored_tensors[trainingpassBiasIdx] != NULL)
                {
                    // monumentum
                    tensor *prev_w_factored = t_mul_const(t_copy(previouspass->stored_tensors[trainingpassWeightsIdx]), monumentum);
                    tensor *prev_b_factored = t_mul_const(t_copy(previouspass->stored_tensors[trainingpassBiasIdx]), monumentum);
                    t_elem_add(t_mul_const(updateWeights, 1 - monumentum), prev_w_factored);
                    t_elem_add(t_mul_const(updateBias, 1 - monumentum), prev_b_factored);
                    t_free(prev_w_factored);
                    t_free(prev_b_factored);
                }

                t_mul_const(updateWeights, batchSizeFactor);
                t_mul_const(updateBias, batchSizeFactor);

                // update stored_tensors
                if (stored_tensors[trainingpassWeightsIdx] == NULL)
                    stored_tensors[trainingpassWeightsIdx] = t_copy(updateWeights);
                else
                    t_elem_add(stored_tensors[trainingpassWeightsIdx], updateWeights);

                if (stored_tensors[trainingpassBiasIdx] == NULL)
                    stored_tensors[trainingpassBiasIdx] = t_copy(updateBias);
                else
                    t_elem_add(stored_tensors[trainingpassBiasIdx], updateBias);

                t_mul_const(updateWeights, learningRate);
                t_mul_const(updateBias, learningRate);

                seq->layers[l]->update(
                    seq->layers[l]->layerProps,
                    updateWeights,
                    updateBias);

                t_free(updateBias);
                t_free(updateWeights);
            }
        }
        for (int l = 0; l < seq->numLayers; l++)
        {
            forwardstate_free(&forwardstates[t][l]);
            backwardstate_free(localbackwardstates[l]);
        }
        mm_free(localbackwardstates);
        mm_free(forwardstates[t]);
    }
    mm_free(forwardstates);
    /**
     * Prediction pass
     */
    for (int t = 0; t < batchSize; t++)
    {
        predictions[t] = seqmodel_predict(seq, inputs[t]);
    }

    struct trainingpass *tp = trainingpass_init(lossFn(batchSize, predictions, truths), stored_tensors, seq->numLayers * 2);

    // Free predictions
    for (int t = 0; t < batchSize; t++)
        t_free(predictions[t]);

    return tp;
}