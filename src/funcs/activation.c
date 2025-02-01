#include <stdio.h>
#include <math.h>

#include "core.h"
#include "funcs.h"

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
    else
    {
        perror("Not existent (or feel free to implement the dirac delta function).");
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
    else
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = v[i] * (1 - v[i]);
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
    else
    {
        for (int i = 0; i < n; i++)
        {
            if (v[i] < 0)
            {
                o[i] = 0;
            }
            else
            {
                o[i] = 1;
            }
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
            o[i] = tanh(v[i]);
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            o[i] = 1 - (tanh(v[i]) * tanh(v[i]));
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
    else
    {
        for (int i = 0; i < n; i++)
        {
            param_t s = 1.0 / (1.0 + exp(-v[i]));
            o[i] = s * (1 - s);
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
    else
    {
        for (int i = 0; i < n; i++)
        {
            // sigmoid(v) * (1-sigmoid(v))
            param_t s = 1.0 / (1.0 + exp(-v[i]));
            o[i] = s * (1 - s);
        }
    }
    return o;
}