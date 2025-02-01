#pragma once

#include "core.h"

#define ACTIVATION_FORWARD 0
#define ACTIVATION_DERIVATIVE 1

typedef vec activationfunc(vec value, int n, int activationMode);

activationfunc av_relu;
activationfunc av_logistic;
activationfunc av_tanh;
activationfunc av_softmax;
activationfunc av_heaviside;
activationfunc av_sigmoid;

typedef param_t lossfunc(int numExamples, int vecSize, vec predictions[numExamples], vec truths[numExamples]);

lossfunc loss_mse;
lossfunc loss_abssum;