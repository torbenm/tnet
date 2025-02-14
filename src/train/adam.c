#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"
#include "train.h"
#include "funcs.h"

struct optimizer *opt_adam_init(param_t alpha, param_t beta1, param_t beta2, lossfunc *lossFn)
{
    struct optimizer *o = mm_alloc(sizeof(struct optimizer));
    o->numParams = 3;
    o->params = mm_alloc(o->numParams * sizeof(param_t));
    o->params[0] = alpha;
    o->params[1] = beta1;
    o->params[2] = beta2;
    o->run_opt = opt_adam;
    o->lossFn = lossFn;
    return o;
}

struct trainingpass *opt_adam(struct seqmodel *seq, param_t *params, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn, struct trainingpass *previouspass, int trainingPassNum)
{
    tensor *predictions[batchSize];

    param_t alpha = params[0];
    param_t beta1 = params[1];
    param_t beta2 = params[2];
    param_t batchSizeFactor = 1.0 / (param_t)batchSize;

    struct tensor **stored_tensors = mm_calloc((seq->numLayers * 4), sizeof(struct tensor *));

    for (int t = 0; t < batchSize; t++)
    {

        struct forwardstate *forwardstates = opt_forwardpropagate(seq, inputs[t], &predictions[t]);
        struct backwardstate **localbackwardstates = opt_backwardpropagate(seq, predictions[t], truths[t], forwardstates);
        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {
            // Only able to update weights if we have a backwardstate.
            // Some layers don't provide us one
            if (localbackwardstates[l] != NULL && localbackwardstates[l]->weightGradients != NULL && localbackwardstates[l]->biasGradients != NULL)
            {
                tensor *momentumWeights = t_copy(localbackwardstates[l]->weightGradients);
                tensor *momentumBias = t_copy(localbackwardstates[l]->biasGradients);
                tensor *velocityWeights = t_copy(localbackwardstates[l]->weightGradients);
                tensor *velocityBias = t_copy(localbackwardstates[l]->biasGradients);

                int momentumWeightsIdx = l * 4;
                int momentumBiasIdx = momentumWeightsIdx + 1;
                int velocityWeightsIdx = momentumWeightsIdx + 2;
                int velocityBiasIdx = momentumWeightsIdx + 3;

                /**
                 * 1. Momentum
                 */
                tensor *prev_w_factored;
                tensor *prev_b_factored;
                if (previouspass != NULL && previouspass->stored_tensors[momentumWeightsIdx] != NULL && previouspass->stored_tensors[momentumBiasIdx] != NULL)
                {
                    prev_w_factored = t_mul_const(t_copy(previouspass->stored_tensors[momentumWeightsIdx]), beta1);
                    prev_b_factored = t_mul_const(t_copy(previouspass->stored_tensors[momentumBiasIdx]), beta1);
                }
                else
                {
                    // zeroes else
                    prev_w_factored = t_alloc(momentumWeights->ndim, momentumWeights->shape);
                    prev_b_factored = t_alloc(momentumBias->ndim, momentumBias->shape);
                }

                t_elem_add(t_mul_const(momentumWeights, 1 - beta1), prev_w_factored);
                t_elem_add(t_mul_const(momentumBias, 1 - beta1), prev_b_factored);
                t_free(prev_w_factored);
                t_free(prev_b_factored);

                t_mul_const(momentumWeights, batchSizeFactor);
                t_mul_const(momentumBias, batchSizeFactor);

                // update stored_tensors
                t_copy_or_add(&stored_tensors[momentumWeightsIdx], momentumWeights);
                t_copy_or_add(&stored_tensors[momentumBiasIdx], momentumBias);

                // momentum bias corrected
                // m / (1 - beta1**(t + 1))
                t_div_const(momentumWeights, (1 - pow(beta1, trainingPassNum + 1)));
                t_div_const(momentumBias, (1 - pow(beta1, trainingPassNum + 1)));

                /**
                 * Velocity
                 */

                if (previouspass != NULL && previouspass->stored_tensors[velocityWeightsIdx] != NULL && previouspass->stored_tensors[velocityBiasIdx] != NULL)
                {
                    prev_w_factored = t_mul_const(t_copy(previouspass->stored_tensors[velocityWeightsIdx]), beta2);
                    prev_b_factored = t_mul_const(t_copy(previouspass->stored_tensors[velocityBiasIdx]), beta2);
                }
                else
                {
                    // zeroes else
                    prev_w_factored = t_alloc(velocityWeights->ndim, velocityWeights->shape);
                    prev_b_factored = t_alloc(velocityBias->ndim, velocityBias->shape);
                }
                // velocity
                t_elem_add(t_mul_const(t_pow_const(velocityWeights, 2), 1 - beta2), prev_w_factored);
                t_elem_add(t_mul_const(t_pow_const(velocityBias, 2), 1 - beta2), prev_b_factored);
                t_free(prev_w_factored);
                t_free(prev_b_factored);

                t_mul_const(velocityWeights, batchSizeFactor);
                t_mul_const(velocityBias, batchSizeFactor);

                // update stored_tensors
                t_copy_or_add(&stored_tensors[velocityWeightsIdx], velocityWeights);
                t_copy_or_add(&stored_tensors[velocityBiasIdx], velocityBias);

                // velocity bias corrected
                // m / (1 - beta1**(t + 1))
                t_div_const(velocityWeights, (1 - pow(beta2, trainingPassNum + 1)));
                t_div_const(velocityBias, (1 - pow(beta2, trainingPassNum + 1)));

                tensor *updateWeights = t_mul_const(t_elem_div(momentumWeights, t_add_const(t_apply(velocityWeights, sqrt), EPSILON)), alpha);
                tensor *updateBias = t_mul_const(t_elem_div(momentumBias, t_add_const(t_apply(velocityBias, sqrt), EPSILON)), alpha);

                seq->layers[l]->update(
                    seq->layers[l]->layerProps,
                    updateWeights,
                    updateBias);

                t_free(updateBias);    // frees momentumBias as well
                t_free(updateWeights); // frees momentumWeights as well
                t_free(velocityBias);
                t_free(velocityWeights);
            }
        }
    }

    param_t loss = seqmodel_calculate_loss(seq, batchSize, inputs, truths, lossFn);
    struct trainingpass *tp = trainingpass_init(loss, stored_tensors, seq->numLayers * 4);

    return tp;
}