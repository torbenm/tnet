#include <math.h>

#include "optimizer.h"
#include "tnet.h"

void opt_gradient_descent(int maxIter, param_t learningRate, param_t costThreshold, param_t diffThreshold, struct params *p)
{
    int converged = 0;
    int iter = 0;

    param_t current_cost = 1000000;
    param_t prev_cost = 0;

    while (!converged && iter < maxIter)
    {
        prev_cost = current_cost;

        converged = fabs(prev_cost - current_cost) < diffThreshold || current_cost < costThreshold;
        iter++;
    }
}