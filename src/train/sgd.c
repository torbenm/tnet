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

void opt_mark(struct optimizer *o)
{
    mm_mark(o);
    mm_mark(o->params);
}

struct trainingpass *opt_sgd(struct seqmodel *seq, param_t *params, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn, struct trainingpass *previouspass, int trainingPassNum)
{

    tensor *predictions[batchSize];

    param_t learningRate = params[0];
    param_t monumentum = params[1];

    /**
     * Backwardpass
     */

    struct tensor **stored_tensors = mm_calloc((seq->numLayers * 2), sizeof(struct tensor *));

    for (int t = 0; t < batchSize; t++)
    {

        struct forwardstate *forwardstates = opt_forwardpropagate(seq, inputs[t], &predictions[t]);
        struct backwardstate **localbackwardstates = opt_backwardpropagate(seq, predictions[t], truths[t], forwardstates);

        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {

            if (localbackwardstates[l] != NULL && localbackwardstates[l]->weightGradients != NULL && localbackwardstates[l]->biasGradients != NULL)
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
    }

    param_t loss = seqmodel_calculate_loss(seq, batchSize, inputs, truths, lossFn);
    struct trainingpass *tp = trainingpass_init(loss, stored_tensors, seq->numLayers * 2);

    return tp;
}