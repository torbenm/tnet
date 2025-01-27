#include "tnet.h"

typedef param_t activationfunc(param_t value);

activationfunc av_relu;
activationfunc av_logistic;
activationfunc av_tanh;
activationfunc av_heaviside;