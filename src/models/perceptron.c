#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "models.h"
#include "funcs.h"

#define LEARNING_RATE 0.1

vec perceptron_forward(struct perceptron *p, vec vals)
{
    vec r = vec_elem_mul(p->weights, vals, p->numWeights);
    vec sum = vec_from_single(p->bias + vec_collapse_sum(r, p->numWeights));
    free(r);
    vec res = p->activationFn(sum, 1, FUNCS_NORMAL);
    free(sum);
    return res;
}

int perceptron_train(struct perceptron *p, int maxIter, int numTVals, mat tvals, vec truth, param_t learningRate)
{
    param_t errSum = 1.0;
    int iter = 0;
    while (errSum > 0.0 && iter < maxIter)
    {
        errSum = 0.0;
        for (int i = 0; i < numTVals; i++)
        {
            vec res = perceptron_forward(p, tvals[i]);
            param_t error = truth[i] - res[0];
            errSum += fabs(error);
            vec deltaW = vec_mul_const(tvals[i], learningRate * error, p->numWeights);
            vec newWeights = vec_elem_add(p->weights, deltaW, p->numWeights);
            free(p->weights);
            free(deltaW);
            p->weights = newWeights;
            p->bias = p->bias + learningRate * error;
        }
        printf("Iter %i: error=%.3f\n", iter, errSum);
        iter++;
    }
    return iter;
}

struct perceptron *perceptron_init(int numWeights)
{
    struct perceptron *p = calloc(1, sizeof(struct perceptron));
    p->weights = vec_alloc_rand(numWeights);
    p->numWeights = numWeights;
    p->bias = prand();
    p->activationFn = av_heaviside;
    return p;
}

void perceptron_free(struct perceptron *p)
{
    free(p->weights);
    free(p);
}
