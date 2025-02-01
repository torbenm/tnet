#pragma once

#include "core.h"

#define FUNCS_NORMAL 0
#define FUNCS_DERIVATIVE 1

typedef vec activationfunc(vec value, int n, int mode);

activationfunc av_relu;
activationfunc av_logistic;
activationfunc av_tanh;
activationfunc av_softmax;
activationfunc av_heaviside;
activationfunc av_sigmoid;

typedef param_t lossfunc(int numExamples, int vecSize, vec predictions[numExamples], vec truths[numExamples], int mode);

lossfunc loss_mse;
lossfunc loss_abssum;