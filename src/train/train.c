#include <stdio.h>
#include <math.h>

#include "models.h"
#include "train.h"
#include "core.h"

void train(struct seqmodel *seq, int numExamples, vec inputs[numExamples], vec truths[numExamples], int maxIter, struct optimizer *opt, param_t diffThreshold, param_t lossThreshold)
{
    int converged = 0;
    int iter = 0;

    param_t current_loss = 1000000;
    param_t prev_loss = 0;

    while (!converged && iter < maxIter)
    {
        prev_loss = current_loss;

        current_loss = opt->run_opt(seq, opt->params, numExamples, inputs, truths, opt->lossFn);
        printf("Iteration %i: loss=%.4f\n", iter, current_loss);

        // Check whether the network is converging...
        converged = fabs(prev_loss - current_loss) < diffThreshold || current_loss < lossThreshold;

        iter++;
    }
}