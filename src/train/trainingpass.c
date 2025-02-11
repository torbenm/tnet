#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "train.h"

struct trainingpass *trainingpass_init(
    param_t loss,
    tensor **storedTensors,
    int numStoredTensors)
{
    struct trainingpass *tp = mm_alloc(sizeof(struct trainingpass));
    tp->loss = loss;
    tp->numStoredTensors = numStoredTensors;
    tp->stored_tensors = storedTensors;
    return tp;
}

void trainingpass_mark(struct trainingpass *tp)
{
    mm_mark(tp);
    mm_mark(tp->stored_tensors);
    for (int l = 0; l < tp->numStoredTensors; l++)
    {
        t_mark(tp->stored_tensors[l]);
    }
}

void trainingpass_free(struct trainingpass *tp)
{
    for (int l = 0; l < tp->numStoredTensors; l++)
    {
        if (tp->stored_tensors[l] != NULL)
            t_free(tp->stored_tensors[l]);
    }
    mm_free(tp);
}

void trainingpass_lock(struct trainingpass *tp)
{
    for (int l = 0; l < tp->numStoredTensors; l++)
    {
        if (tp->stored_tensors[l] != NULL)
            t_lock(tp->stored_tensors[l]);
    }
}