#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "models.h"
#include "train.h"

#define LEARNING_RATE 0.1

void perceptron_tensor_forward_print(struct perceptron_tensor *p, int batchSize, tensor inputs[batchSize], tensor truths[batchSize])
{
    // for (int i = 0; i < batchSize; i++)
    // {
    //     if (i > 0)
    //         printf(";\t");
    //     // vec_print(vals, p->numWeights); -- tensor_print()
    //     printf("=");
    //     vec res = perceptron_tensor_forward(p, inputs[i]);
    //     printf("%.3f(%.3f)", res[0], inputs[i]);
    // }
    // printf("\n");
}

void perceptron_tensor_example_AND()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t expected[4] = {0.0, 0.0, 0.0, 1.0};

    tensor *t_values = t_from_2dim_array(4, 2, values);
    t_print(t_values);
    struct perceptron_tensor *p = perceptron_tensor_init(2);

    // int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    // printf("After %i Iterations: ", usedIter);
    // perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
}

void perceptron_tensor_example_OR()
{
    // param_t values[4][2] = {{0.0, 0.0},
    //                         {0.0, 1.0},
    //                         {1.0, 0.0},
    //                         {1.0, 1.0}};
    // param_t expected[4] = {0.0, 1.0, 1.0, 1.0};

    // struct perceptron *p = perceptron_init(2);
    // int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    // printf("After %i Iterations: ", usedIter);
    // perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
}

void perceptron_tensor_example_XOR()
{

    // param_t values[4][2] = {{0.0, 0.0},
    //                         {0.0, 1.0},
    //                         {1.0, 0.0},
    //                         {1.0, 1.0}};
    // param_t expected[4] = {0.0, 1.0, 1.0, 0.0};

    // struct perceptron *p = perceptron_init(2);
    // int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    // printf("After %i Iterations: ", usedIter);
    // perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
    // printf("XOR is not supported by Perceptrons! Thus, the result will not be correct.");
}
