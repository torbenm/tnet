#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "models.h"
#include "funcs.h"

#define LEARNING_RATE 0.1

tensor *perceptron_forward(struct perceptron *p, tensor *inputs)
{
    tensor *weights_x_inputs = t_elem_mul(t_copy(p->weights), inputs);
    tensor *plus_bias = t_elem_add(t_collapse_sum(weights_x_inputs, -1), p->bias);

    // t_collapse_sum creates new tensor, so freeing the previously created weights_x_inputs
    t_free(weights_x_inputs);

    return p->activationFn(plus_bias, FUNCS_NORMAL);
}

int perceptron_train(struct perceptron *p, int maxIter, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], param_t learningRate)
{
    tensor *errSum = t_alloc_single();
    t_init_const(errSum, 1.0); // setting to 1.0 so that we enter the first loop
    int iter = 0;
    while (errSum->v[0] > 0.0 && iter < maxIter)
    {
        errSum->v[0] = 0.0;
        for (int i = 0; i < batchSize; i++)
        {
            // error = pred - truth
            tensor *pred = perceptron_forward(p, inputs[i]);
            tensor *error = t_elem_sub(t_copy(truths[i]), pred);

            // error_sum = |error|
            tensor *absError = t_apply(t_copy(error), fabs);
            t_elem_add(errSum, absError);

            // weights = weights + input * error * learningRate
            tensor *learningRateError = t_mul_const(error, learningRate);
            tensor *deltaWeights = t_elem_mul(t_copy(inputs[i]), learningRateError);
            t_elem_add(p->weights, deltaWeights);

            // bias = bias + error * learningRate
            t_elem_add(p->bias, learningRateError);

            // Free all alloced
            t_free(absError);
            t_free(deltaWeights);
            t_free(learningRateError); // also frees 'error'
            t_free(pred);
        }
        printf("Iter %i: error=", iter);
        t_print(errSum);
        printf("\n");
        iter++;
    }
    return iter;
}

struct perceptron *perceptron_init(int numWeights)
{
    struct perceptron *p = calloc(1, sizeof(struct perceptron));
    const int weights_shape[] = {numWeights};
    p->weights = t_alloc(1, weights_shape);
    t_init_rand(p->weights);

    p->bias = t_alloc_single();
    t_init_rand(p->bias);

    p->activationFn = av_heaviside_tensor;
    return p;
}

void perceptron_free(struct perceptron *p)
{
    t_free(p->bias);
    t_free(p->weights);
    free(p);
}
