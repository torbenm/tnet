#pragma once

#include "core.h"

#define FUNCS_NORMAL 0
#define FUNCS_DERIVATIVE 1

typedef tensor *activationfunc(tensor *t, int mode);

activationfunc av_relu;
activationfunc av_logistic;
activationfunc av_tanh;
activationfunc av_heaviside;
activationfunc av_sigmoid;
activationfunc av_identity;

typedef param_t loss_forward(tensor *p, tensor *t);
typedef tensor *loss_backward(tensor *p, tensor *t);

typedef struct loss
{
    loss_forward *forward;
    loss_backward *backward;
} loss;

param_t loss_over_batch(loss *l, int batchSize, tensor *predictions[batchSize], tensor *truths[batchSize]);
loss *loss_mse();
loss *loss_binary_cross_entropy();