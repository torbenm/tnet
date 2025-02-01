#include <stdio.h>

#include "core.h"
#include "models.h"
#include "train.h"

#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 100000
#define LEARNING_RATE 0.1

void mlp_forward_print(struct mlp *p, int numVals, vec values[numVals], vec truth[numVals])
{
    vec predictions[numVals];
    struct intermediate ims[p->numLayers + 1];
    for (int i = 0; i < numVals; i++)
    {
        if (i > 0)
            printf(";\t");
        for (int w = 0; w < p->numInputs; w++)
        {
            if (w > 0)
                printf(",");
            printf("%.3f", values[i][w]);
        }
        printf("=");
        predictions[i] = mlp_forward(p, values[i], ims);
        vec_print(predictions[i], p->numOutputs);
        printf("(");
        vec_print(truth[i], p->numOutputs);
        printf(")");
    }
    param_t loss = p->lossFn(numVals, 1, predictions, truth);
    printf(". Loss=%.3f\n", loss);
}

void mlp_example_XOR()
{
    const int numLayers = 1;
    int numParamsPerLayer[3] = {2};

    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

    struct mlp *p = mlp_init(2, 2, numLayers, numParamsPerLayer);
    mlp_forward_print(p, 4, vec_array_from_array_of_arrays(4, 2, values), vec_array_from_array_of_arrays(4, 2, truths));
    opt_gradient_descent(MAX_ITER, LEARNING_RATE, COST_THRESHOLD, DIFF_THRESHOLD, mat_from_array(4, 2, values), 4, mat_from_array(4, 2, truths), p);
    mlp_forward_print(p, 4, vec_array_from_array_of_arrays(4, 2, values), vec_array_from_array_of_arrays(4, 2, truths));
}