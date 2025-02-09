#include <math.h>

#include "core.h"
#include "funcs.h"

param_t loss_mse(int numExamples, tensor *predictions[numExamples], tensor *truths[numExamples])
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        // sum += (truth - pred)^2 / num_nodes
        tensor *err = t_copy(truths[i]);
        t_elem_sub(err, predictions[i]);
        t_elem_mul(err, err);
        sum += t_collapse_mean_all(err);
        t_free(err);
    }
    return sum / (param_t)numExamples;
}

param_t loss_abssum(int numExamples, tensor *predictions[numExamples], tensor *truths[numExamples])
{
    param_t sum = 0;
    for (int i = 0; i < numExamples; i++)
    {
        // sum += |(truth - pred)| / num_nodes
        tensor *err = t_copy(truths[i]);
        t_elem_sub(err, predictions[i]);
        t_apply(err, fabs);
        sum += t_collapse_mean_all(err);
        t_free(err);
    }
    return sum / (param_t)numExamples;
}