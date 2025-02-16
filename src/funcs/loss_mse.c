#include <stdio.h>
#include "core.h"
#include "funcs.h"

param_t loss_mse_forward(tensor *p, tensor *t)
{
    tensor *err = t_copy(t);
    t_elem_sub(err, p);
    t_elem_mul(err, err);
    param_t r = t_collapse_mean_all(err);
    t_free(err);
    return r;
}

tensor *loss_mse_backward(tensor *p, tensor *t)
{
    return t_mul_const(t_elem_sub(t_copy(p), t), 2);
}

loss *loss_mse()
{
    loss *l = mm_alloc(sizeof(loss));
    l->forward = loss_mse_forward;
    l->backward = loss_mse_backward;
    return l;
}