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

    struct trainingpass *prev_pass = NULL;

    while (!converged && iter < maxIter)
    {
        prev_loss = current_loss;

        struct trainingpass *next_pass = opt->run_opt(seq, opt->params, numExamples, inputs, truths, opt->lossFn, prev_pass);
        current_loss = next_pass->loss;
        printf("Iteration %i: loss=%.4f\n", iter, current_loss);
        if (prev_pass != NULL)
            trainingpass_free(prev_pass);
        prev_pass = next_pass;

        // Check whether the network is converging...
        converged = fabs(prev_loss - current_loss) < diffThreshold || current_loss < lossThreshold;

        iter++;
    }
}