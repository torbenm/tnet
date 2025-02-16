#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "core.h"
#include "funcs.h"
#include "models.h"
#include "train.h"

#define TEST_FAILURE 0
#define TEST_SUCCESS 1
#define E 0.000001

void __check_gradient_for_tensor(struct seqmodel *s, tensor *t, tensor *grad, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], param_t *numerator, param_t *denominator, loss *loss)
{
    for (int i = 0; i < t->_v_size; i++)
    {
        param_t original = t->v[i];
        // plus
        t->v[i] = original + E;
        param_t lossplus = seqmodel_calculate_loss(s, batchSize, inputs, truths, loss);

        // minus
        t->v[i] = original - E; // remove twice to account for above
        param_t lossminus = seqmodel_calculate_loss(s, batchSize, inputs, truths, loss);
        t->v[i] = original; // undo

        param_t ag = (lossplus - lossminus) / (2 * E);
        param_t cg = grad->v[i];

        param_t num = cg - ag;
        printf("| | |-%i: ag=%f; cg=%f\n", i, ag, grad->v[i]);
        *numerator += pow(pow(num, 2.0), 0.5);
        *denominator += pow(pow(ag, 2.0), 0.5) + pow(pow(cg, 2.0), 0.5);
    }
}

void check_gradients(struct seqmodel *s, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], loss *loss)
{
    param_t numerator = 0;
    param_t denominator = 0;

    tensor *prediction;
    tensor **totalWeightGradients = mm_calloc(s->numLayers, sizeof(tensor *));
    tensor **totalBiasGradients = mm_calloc(s->numLayers, sizeof(tensor *));

    opt_fowardbackwardpass(s, batchSize, inputs, truths, &totalWeightGradients, &totalBiasGradients, loss);

    for (int l = 0; l < s->numLayers; l++)
    {
        printf("|-Layer %i\n", l);
        if (totalWeightGradients[l] != NULL && totalBiasGradients[l] != NULL)
        {
            printf("| |-For Weights:\n");
            // check weights
            __check_gradient_for_tensor(s, s->layers[l]->weights, totalWeightGradients[l], batchSize, inputs, truths, &numerator, &denominator, loss);
            printf("| |-For Bias:\n");
            // check bias
            __check_gradient_for_tensor(s, s->layers[l]->bias, totalBiasGradients[l], batchSize, inputs, truths, &numerator, &denominator, loss);
        }
        else
        {
            printf("| |-No gradients.\n");
        }
    }

    param_t diff = numerator / denominator;
    printf("Total difference: %f", diff);

    if (diff < 1e-7)
        printf(" ✅");
    else
        printf(" ❌");
    printf("\n");
}