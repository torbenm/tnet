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

struct trainingpass *opt_sgd(struct seqmodel *seq, param_t *params, int batchSize, vec inputs[batchSize], vec truths[batchSize], lossfunc *lossFn, struct trainingpass *previouspass)
{
    setbuf(stdout, 0);

    struct forwardstate **forwardstates = malloc(batchSize * sizeof(struct forwardstate *));
    vec predictions[batchSize];

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
        }
        predictions[t] = forwardstates[t][seq->numLayers - 1].activations;
    }

    /**
     * Backwardpass
     */

    struct backwardstate *globalbackwardstates = malloc(seq->numLayers * sizeof(struct backwardstate));

    for (int t = 0; t < batchSize; t++)
    {
        vec nextDelta = vec_elem_sub(predictions[t], truths[t], seq->layers[seq->numLayers - 1]->numOutputs);
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
                learningRate,
                (l == (seq->numLayers - 1)));

            if (localbackwardstates[l] != NULL)
            {
                nextDelta = localbackwardstates[l]->smallDelta;
            }
        }
        // Apply deltas pass
        for (int l = 0; l < seq->numLayers; l++)
        {

            if (monumentum > 0 && previouspass != NULL && localbackwardstates[l] != NULL && &previouspass->backwardstates[l] != NULL)
            {
                // Applying monumentum
                int numNodes = localbackwardstates[l]->numNodes;
                int numInputs = localbackwardstates[l]->numInputs;
                // previous update weights * monumentum
                mat mon_w_prev = mat_mul_const(previouspass->backwardstates[l].weightGradients, monumentum, numNodes, numInputs);
                vec mon_b_prev = vec_mul_const(previouspass->backwardstates[l].biasGradients, monumentum, numNodes);

                mat mon_w_curr = mat_mul_const(localbackwardstates[l]->weightGradients, 1 - monumentum, numNodes, numInputs);
                mat mon_b_curr = vec_mul_const(localbackwardstates[l]->biasGradients, 1 - monumentum, numNodes);

                mat new_w = mat_elem_add(mon_w_prev, mon_w_curr, numNodes, numInputs);
                vec new_b = vec_elem_add(mon_b_prev, mon_b_curr, numNodes);

                mat_free(localbackwardstates[l]->weightGradients, numNodes);
                vec_free(localbackwardstates[l]->biasGradients);

                mat_free(mon_w_prev, numNodes);
                vec_free(mon_b_prev);

                mat_free(mon_w_curr, numNodes);
                vec_free(mon_b_curr);

                localbackwardstates[l]->weightGradients = new_w;
                localbackwardstates[l]->biasGradients = new_b;
            }

            param_t batchSizeFactor = 1.0 / (param_t)batchSize;

            seq->layers[l]->update(
                seq->layers[l]->layerProps,
                localbackwardstates[l],
                batchSizeFactor * learningRate);

            // updating the global backwardstate
            backwardstate_incorporate(&globalbackwardstates[l], localbackwardstates[l], batchSizeFactor);

            // free memory - not needed any longer
            backwardstate_free(localbackwardstates[l]);
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
    return trainingpass_init(lossFn(batchSize, 1, predictions, truths), globalbackwardstates, seq->numLayers);
}