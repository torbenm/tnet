#include <stdio.h>
#include <math.h>

#include "models.h"
#include "train.h"
#include "core.h"

void __train_mark_and_sweep(struct seqmodel *seq, int numExamples, tensor *inputs[numExamples], tensor *truths[numExamples], struct optimizer *opt, struct trainingpass *tp)
{
    // mark
    seqmodel_mark(seq);
    mm_mark(inputs);
    mm_mark(truths);
    for (int i = 0; i < numExamples; i++)
    {
        t_mark(inputs[i]);
        t_mark(truths[i]);
    }
    if (tp != NULL)
        trainingpass_mark(tp);
    opt_mark(opt);

    // sweep
    mm_sweep();
    mm_unmark_all();
}

void train(struct seqmodel *seq, int numExamples, tensor *inputs[numExamples], tensor *truths[numExamples], int maxIter, struct optimizer *opt, param_t diffThreshold, param_t lossThreshold)
{
    int converged = 0;
    int iter = 0;

    param_t current_loss = 1000000;
    param_t prev_loss = 0;

    struct trainingpass *prev_pass = NULL;

    while (!converged && iter < maxIter)
    {
        __train_mark_and_sweep(seq, numExamples, inputs, truths, opt, prev_pass);
        prev_loss = current_loss;
        struct trainingpass *next_pass = opt->run_opt(seq, opt->params, numExamples, inputs, truths, opt->lossFn, prev_pass, iter);
        current_loss = next_pass->loss;
        printf("Iteration %i: loss=%.4f", iter, current_loss);
        if (iter == 0)
            printf("\n");
        else
        {
            printf("\r");
        }
        if (prev_pass != NULL)
            trainingpass_free(prev_pass);
        prev_pass = next_pass;
        trainingpass_lock(prev_pass);

        // // Check whether the network is converging...
        converged = fabs(prev_loss - current_loss) < diffThreshold || current_loss < lossThreshold;
        iter++;
    }
    printf("\n");
    trainingpass_free(prev_pass);
}