#include <stdio.h>
#include <math.h>

#include "models.h"
#include "train.h"
#include "core.h"

void __train_dodge_and_wipe(struct seqmodel *seq, int numExamples, tensor *inputs[numExamples], tensor *truths[numExamples], struct optimizer *opt, struct trainingpass *tp)
{
    // dodge
    seqmodel_dodge(seq);
    for (int i = 0; i < numExamples; i++)
    {
        t_dodge(inputs[i]);
        t_dodge(truths[i]);
    }
    if (tp != NULL)
        trainingpass_dodge(tp);
    optimizer_dodge(opt);

    // wipe
    mm_wipe();
    mm_undodge_all();
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
        __train_dodge_and_wipe(seq, numExamples, inputs, truths, opt, prev_pass);
        prev_loss = current_loss;

        struct trainingpass *next_pass = opt->run_opt(seq, opt->params, numExamples, inputs, truths, opt->lossFn, prev_pass);
        current_loss = next_pass->loss;
        printf("Iteration %i: loss=%.4f\n", iter, current_loss);
        if (prev_pass != NULL)
            trainingpass_free(prev_pass);
        prev_pass = next_pass;
        trainingpass_lock(prev_pass);

        // // Check whether the network is converging...
        converged = fabs(prev_loss - current_loss) < diffThreshold || current_loss < lossThreshold;

        iter++;
    }
    trainingpass_free(prev_pass);
}