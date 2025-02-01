#include "tnet.h"
#include "mat.h"

#define ACTIVATION_FORWARD 0
#define ACTIVATION_DERIVATIVE 1

typedef vec activationfunc(vec value, int n, int activationMode);

activationfunc av_relu;
activationfunc av_logistic;
activationfunc av_tanh;
activationfunc av_softmax;
activationfunc av_heaviside;