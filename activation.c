#include <math.h>

#include "tnet.h"
#include "activation.h"

param_t av_heaviside(param_t val)
{
    return val >= 0;
}

param_t av_logistic(param_t val)
{
    return 1.0 / (1.0 + exp(-val));
}

param_t av_relu(param_t val)
{
    return fmax(0.0, val);
}

param_t av_tanh(param_t val)
{
    return (exp(val) - exp(-val)) / (exp(val) + exp(-val));
}