#include <stdio.h>
#include <math.h>

#include "core.h"
#include "funcs.h"

tensor *av_heaviside(tensor *t, int activationMode)
{
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = t->v[i] >= 0;
        }
    }
    else
    {
        error("Not existent (or feel free to implement the dirac delta function).");
    }
    return t;
}

tensor *av_logistic(tensor *t, int activationMode)
{
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = 1.0 / (1.0 + exp(-t->v[i]));
        }
    }
    else
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = t->v[i] * (1 - t->v[i]);
        }
    }
    return t;
}

tensor *av_relu(tensor *t, int activationMode)
{
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = fmax(0.0, t->v[i]);
        }
    }
    else
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            if (t->v[i] < 0)
            {
                t->v[i] = 0;
            }
            else
            {
                t->v[i] = 1;
            }
        }
    }
    return t;
}

tensor *av_tanh(tensor *t, int activationMode)
{
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = tanh(t->v[i]);
        }
    }
    else
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = 1 - (tanh(t->v[i]) * tanh(t->v[i]));
        }
    }
    return t;
}

tensor *av_sigmoid(tensor *t, int activationMode)
{
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = 1.0 / (1.0 + exp(-t->v[i]));
        }
    }
    else
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            param_t s = 1.0 / (1.0 + exp(-t->v[i]));
            t->v[i] = s * (1 - s);
        }
    }
    return t;
}

tensor *av_softmax(tensor *t, int activationMode)
{
    // Softmax acts across the whole tensor
    t_assert_not_locked(t);

    if (activationMode == FUNCS_NORMAL)
    {
        param_t sumOfAll = 0;
        for (int i = 0; i < t->_v_size; i++)
            sumOfAll += exp(t->v[i]);

        for (int i = 0; i < t->_v_size; i++)
        {
            t->v[i] = exp(t->v[i]) / sumOfAll;
        }
    }
    else
    {
        for (int i = 0; i < t->_v_size; i++)
        {
            // sigmoid(v) * (1-sigmoid(v))
            param_t s = 1.0 / (1.0 + exp(-1 * t->v[i]));
            t->v[i] = s * (1 - s);
        }
    }
    return t;
}