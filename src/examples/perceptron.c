#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "models.h"
#include "train.h"

#define LEARNING_RATE 0.1

void _perceptron_forward_print(struct perceptron *p, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize])
{
    for (int i = 0; i < batchSize; i++)
    {
        if (i > 0)
            printf(";\n");
        t_print(inputs[i]);
        printf("=");
        tensor *res = perceptron_forward(p, inputs[i]);
        t_print(res);
        printf(" (truth=");
        t_print(truths[i]);
        printf(")");
    }
    printf("\n");
}

void _perceptron_execute(param_t inputs[4][2], param_t truths[4][1])
{
    tensor *t_inputs[4] = {
        t_from_1dim_array(2, inputs[0]),
        t_from_1dim_array(2, inputs[1]),
        t_from_1dim_array(2, inputs[2]),
        t_from_1dim_array(2, inputs[3]),
    };
    tensor *t_truths[4] = {
        t_from_1dim_array(1, truths[0]),
        t_from_1dim_array(1, truths[1]),
        t_from_1dim_array(1, truths[2]),
        t_from_1dim_array(1, truths[3]),
    };

    struct perceptron *p = perceptron_init(2);
    printf("Initial predictions: \n");
    _perceptron_forward_print(p, 4, t_inputs, t_truths);
    printf("\nTraining: \n");
    int usedIter = perceptron_train(p, 100, 4, t_inputs, t_truths, LEARNING_RATE);
    printf("\nAfter %i Iterations:\n", usedIter);
    _perceptron_forward_print(p, 4, t_inputs, t_truths);
}

void perceptron_example_AND()
{
    param_t inputs[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][1] = {{0.0}, {0.0}, {0.0}, {1.0}};
    _perceptron_execute(inputs, truths);
}

void perceptron_example_OR()
{
    param_t inputs[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][1] = {{0.0}, {1.0}, {1.0}, {1.0}};

    _perceptron_execute(inputs, truths);
}

void perceptron_example_XOR()
{

    param_t inputs[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};
    _perceptron_execute(inputs, truths);
    printf("Perceptrons cannot learn XOR! Thus, the result will not be correct.\n");
}
