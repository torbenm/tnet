#include <math.h>
#include <stdio.h>

#include "core.h"
#include "funcs.h"

param_t loss_over_batch(loss *l, int batchSize, tensor *predictions[batchSize], tensor *truths[batchSize])
{
    param_t sum = 0;
    for (int t = 0; t < batchSize; t++)
    {
        sum += l->forward(predictions[t], truths[t]);
    }
    return sum / (param_t)batchSize;
}
