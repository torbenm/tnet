#include <math.h>

#include "tnet.h"
#include "activation.h"
#include "mat.h"

vec av_heaviside(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);
    if (activationMode == ACTIVATION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = v[i] >= 0;
        }
    }
    return o;
}

vec av_logistic(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);

    if (activationMode == ACTIVATION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = 1.0 / (1.0 + exp(-v[i]));
        }
    }
    return o;
}

vec av_relu(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);

    if (activationMode == ACTIVATION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = fmax(0.0, v[i]);
        }
    }
    return o;
}

vec av_tanh(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);

    if (activationMode == ACTIVATION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = (exp(v[i]) - exp(-v[i])) / (exp(v[i]) + exp(-v[i]));
        }
    }
    return o;
}

vec av_sigmoid(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);

    if (activationMode == ACTIVATION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = 1.0 / (1.0 + exp(-v[i]));
        }
    }
    return o;
}

vec av_softmax(vec v, int n, int activationMode)
{
    vec o = vec_alloc(n);

    if (activationMode == ACTIVATION_FORWARD)
    {
        param_t sumOfAll = 0;
        for (int i = 0; i < n; i++)
            sumOfAll += exp(v[i]);

        for (int i = 0; i < n; i++)
        {
            o[i] = exp(v[i]) / sumOfAll;
        }
    }
    return o;
}