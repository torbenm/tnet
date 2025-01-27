#include <math.h>

#include "tnet.h"
#include "loss.h"

param_t loss_mse(int numExamples, param_t predictions[numExamples], param_t truths[numExamples])
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        param_t err = truths[i] - predictions[i];
        sum += err * err;
    }
    return sum / (param_t)numExamples;
}

param_t loss_abssum(int numExamples, param_t predictions[numExamples], param_t truths[numExamples])
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        param_t err = truths[i] - predictions[i];
        sum += fabs(err);
    }
    return sum;
}