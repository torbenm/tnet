#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "activation.h"
#include "tnet.h"
#include "perceptron.h"

#define LEARNING_RATE 0.1

void perceptron_forward_print(struct perceptron *p, int numVals, mat values, vec truth)
{
    for (int i = 0; i < numVals; i++)
    {
        vec vals = vec_from_mat_col(i, p->numWeights, values);
        if (i > 0)
            printf(";\t");

        vec_print(vals, p->numWeights);
        printf("=");
        vec res = perceptron_forward(p, vals);
        printf("%.3f(%.3f)", res[0], truth[i]);
    }
    printf("\n");
}

void perceptron_example_AND()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t expected[4] = {0.0, 0.0, 0.0, 1.0};
    struct perceptron *p = perceptron_init(2);
    int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    printf("After %i Iterations: ", usedIter);
    perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
}

void perceptron_example_OR()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t expected[4] = {0.0, 1.0, 1.0, 1.0};

    struct perceptron *p = perceptron_init(2);
    int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    printf("After %i Iterations: ", usedIter);
    perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
}

void perceptron_example_XOR()
{

    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t expected[4] = {0.0, 1.0, 1.0, 0.0};

    struct perceptron *p = perceptron_init(2);
    int usedIter = perceptron_train(p, 100, 4, mat_from_array(4, 2, values), vec_from_array(4, expected), LEARNING_RATE);
    printf("After %i Iterations: ", usedIter);
    perceptron_forward_print(p, 4, mat_from_array(4, 2, values), vec_from_array(4, expected));
    printf("XOR is not supported by Perceptrons! Thus, the result will not be correct.");
}
