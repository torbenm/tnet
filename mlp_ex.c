#include <stdio.h>
#include "mlp_ex.h"
#include "mlp.h"

void mlp_forward_print(struct mlp *p, int numVals, param_t values[numVals][p->numInputs], param_t truth[numVals])
{
    param_t predictions[numVals];
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
        predictions[i] = mlp_forward(p, values[i])[0];
        printf("%.3f(%.3f)", predictions[i], truth[i]);
    }
    param_t loss = p->lossFn(numVals, predictions, truth);
    printf(". Loss=%.3f\n", loss);
}

void mlp_example_XOR()
{
    const int numLayers = 1;
    int numParamsPerLayer[1] = {1};

    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4] = {0.0, 1.0, 1.0, 0.0};

    struct mlp *p = mlp_init(2, 1, numLayers, numParamsPerLayer);
    mlp_forward_print(p, 4, values, truths);
}