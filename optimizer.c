#include <math.h>

#include "optimizer.h"
#include "tnet.h"
#include "mlp.h"

void opt_gradient_descent(int maxIter, param_t learningRate, param_t costThreshold, param_t diffThreshold, mat trainingValues, int numTrainingValues, struct mlp *p)
{
    int converged = 0;
    int iter = 0;

    param_t current_cost = 1000000;
    param_t prev_cost = 0;

    while (!converged && iter < maxIter)
    {
        prev_cost = current_cost;

        for (int t = 0; t < numTrainingValues; t++)
        {
            vec output = mlp_forward(p, trainingValues[t]);
        }
        // calculate deltas

        // Check whether the network is converging...
        converged = fabs(prev_cost - current_cost) < diffThreshold || current_cost < costThreshold;
        iter++;
    }
}