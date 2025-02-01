#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "optimizer.h"
#include "tnet.h"
#include "mat.h"
#include "intermediate.h"
#include "mlp.h"

void opt_gradient_descent(int maxIter, param_t learningRate, param_t costThreshold, param_t diffThreshold, mat trainingValues, int numTrainingValues, mat truths, struct mlp *p)
{
    int converged = 0;
    int iter = 0;

    param_t current_cost = 1000000;
    param_t prev_cost = 0;

    while (!converged && iter < maxIter)
    {
        prev_cost = current_cost;
        struct intermediate **ims_per_training = malloc(numTrainingValues * sizeof(struct intermediate *));
        vec predictions[numTrainingValues];
        for (int t = 0; t < numTrainingValues; t++)
        {
            ims_per_training[t] = malloc((p->numLayers + 1) * sizeof(struct intermediate));
            // mat_print(p->layers[0]->weights, p->layers[0]->numInputs, p->layers[0]->numNodes);
            // printf("\n");
            predictions[t] = mlp_forward(p, trainingValues[t], ims_per_training[t]);
        }
        current_cost = p->lossFn(numTrainingValues, 1, predictions, truths);
        printf("Iteration %i: loss=%.4f\n", iter, current_cost);
        // calculate deltas
        for (int t = 0; t < numTrainingValues; t++)
        {
            mlp_backward(p, vec_elem_sub(truths[t], predictions[t], p->numInputs), learningRate, ims_per_training[t]);
        }

        // Check whether the network is converging...
        converged = fabs(prev_cost - current_cost) < diffThreshold || current_cost < costThreshold;
        iter++;
    }
}

void opt_adam(int maxIter, param_t learningRate, param_t beta1, param_t beta2, param_t costThreshold, param_t diffThreshold, mat trainingValues, int numTrainingValues, mat truths, struct mlp *p)
{
    int converged = 0;
    int iter = 0;

    param_t current_cost = 1000000;
    param_t prev_cost = 0;

    while (!converged && iter < maxIter)
    {
        prev_cost = current_cost;
        struct intermediate **ims_per_training = malloc(numTrainingValues * sizeof(struct intermediate *));
        vec predictions[numTrainingValues];
        for (int t = 0; t < numTrainingValues; t++)
        {
            ims_per_training[t] = malloc((p->numLayers + 1) * sizeof(struct intermediate));
            // mat_print(p->layers[0]->weights, p->layers[0]->numInputs, p->layers[0]->numNodes);
            // printf("\n");
            predictions[t] = mlp_forward(p, trainingValues[t], ims_per_training[t]);
        }
        current_cost = p->lossFn(numTrainingValues, 1, predictions, truths);
        printf("Iteration %i: loss=%.4f\n", iter, current_cost);
        // calculate deltas
        for (int t = 0; t < numTrainingValues; t++)
        {
            mlp_backward(p, vec_elem_sub(truths[t], predictions[t], p->numInputs), learningRate, ims_per_training[t]);
        }

        // Check whether the network is converging...
        converged = fabs(prev_cost - current_cost) < diffThreshold || current_cost < costThreshold;
        iter++;
    }
}