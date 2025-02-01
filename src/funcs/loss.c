#include <math.h>

#include "core.h"
#include "funcs.h"

param_t loss_mse(int numExamples, int vecSize, vec predictions[numExamples], vec truths[numExamples], int derivative)
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        param_t v_sum = 0;
        for (int v = 0; v < vecSize; v++)
        {
            param_t err = truths[i][v] - predictions[i][v];
            v_sum += err * err;
        }
        sum += v_sum / (param_t)vecSize;
    }
    return sum / (param_t)numExamples;
}

param_t loss_abssum(int numExamples, int vecSize, vec predictions[numExamples], vec truths[numExamples], int derivative)
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        param_t v_sum = 0;
        for (int v = 0; v < vecSize; v++)
        {
            param_t err = truths[i][v] - predictions[i][v];
            v_sum += fabs(err);
        }
        sum += v_sum / (param_t)vecSize;
    }
    return sum / (param_t)numExamples;
}