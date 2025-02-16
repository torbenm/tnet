#include <stdio.h>
#include <math.h>
#include "core.h"
#include "funcs.h"

param_t loss_binary_cross_entropy_forward(tensor *p, tensor *t)
{
    param_t sum = 0;
    // Element-wise operation for BCE: - y * log(p) - (1 - y) * log(1 - p)
    for (int j = 0; j < t->_v_size; j++)
    {
        param_t y = t->v[j];
        param_t y_hat = clip(p->v[j], EPSILON, 1 - EPSILON);

        // Calculate binary cross-entropy component
        sum += -(y * log(y_hat) + (1 - y) * log(1 - y_hat));
    }
    return sum / (param_t)t->_v_size;
}

tensor *loss_binary_cross_entropy_backward(tensor *p, tensor *t)
{
    return t_elem_sub(t_copy(p), t);
}

loss *loss_binary_cross_entropy()
{
    loss *l = mm_alloc(sizeof(loss));
    l->forward = loss_binary_cross_entropy_forward;
    l->backward = loss_binary_cross_entropy_backward;
    return l;
}