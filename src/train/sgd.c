#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"
#include "train.h"
#include "funcs.h"

struct optimizer *opt_sgd_init(param_t learningRate, param_t monumentum, loss *loss)
{
    optimizer *o = mm_alloc(sizeof(struct optimizer));
    o->name = "SGD";
    o->numParams = 2;
    o->params = mm_alloc(o->numParams * sizeof(param_t));
    o->params[0] = learningRate;
    o->params[1] = monumentum;
    o->run_opt = opt_sgd;
    o->loss = loss;
    return o;
}

struct trainingpass *opt_sgd(struct seqmodel *seq, param_t *params, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], loss *loss, struct trainingpass *previouspass, int trainingPassNum)
{

    param_t learningRate = params[0];
    param_t monumentum = params[1];

    /**
     * Backwardpass
     */

    tensor **stored_tensors = mm_calloc((seq->numLayers * 2), sizeof(tensor *));
    tensor **totalWeightGradients = mm_calloc(seq->numLayers, sizeof(tensor *));
    tensor **totalBiasGradients = mm_calloc(seq->numLayers, sizeof(tensor *));

    opt_fowardbackwardpass(seq, batchSize, inputs, truths, &totalWeightGradients, &totalBiasGradients, loss);

    // Apply deltas pass
    for (int l = 0; l < seq->numLayers; l++)
    {
        if (totalWeightGradients[l] != NULL && totalBiasGradients[l] != NULL)
        {
            tensor *updateWeights = t_copy(totalWeightGradients[l]);
            tensor *updateBias = t_copy(totalBiasGradients[l]);

            int trainingpassWeightsIdx = l * 2;
            int trainingpassBiasIdx = l * 2 + 1;

            tensor *prev_w_factored;
            tensor *prev_b_factored;
            if (previouspass != NULL && previouspass->stored_tensors[trainingpassWeightsIdx] != NULL && previouspass->stored_tensors[trainingpassBiasIdx] != NULL)
            {
                // monumentum
                prev_w_factored = t_mul_const(t_copy(previouspass->stored_tensors[trainingpassWeightsIdx]), monumentum);
                prev_b_factored = t_mul_const(t_copy(previouspass->stored_tensors[trainingpassBiasIdx]), monumentum);
            }
            else
            {
                prev_w_factored = t_alloc(updateWeights->ndim, updateWeights->shape);
                prev_b_factored = t_alloc(updateBias->ndim, updateBias->shape);
            }

            t_elem_add(t_mul_const(updateWeights, 1.0 - monumentum), prev_w_factored);
            t_elem_add(t_mul_const(updateBias, 1.0 - monumentum), prev_b_factored);
            t_free(prev_w_factored);
            t_free(prev_b_factored);

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
                seq->layers[l],
                updateWeights,
                updateBias);

            t_free(updateBias);
            t_free(updateWeights);
        }
    }

    param_t pass_loss = seqmodel_calculate_loss(seq, batchSize, inputs, truths, loss);
    struct trainingpass *tp = trainingpass_init(pass_loss, stored_tensors, seq->numLayers * 2);

    return tp;
}