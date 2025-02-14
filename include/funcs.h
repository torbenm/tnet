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

typedef param_t lossfunc(int numExamples, tensor *predictions[numExamples], tensor *truths[numExamples]);

lossfunc loss_mse;
lossfunc loss_abssum;