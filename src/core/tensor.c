#include <stdio.h>
#include <stdlib.h>
#include "core.h"

void _assert_shape_equals(tensor *t1, tensor *t2)
{
    if (t1->ndim != t2->ndim)
    {
        perror("Shapes are not equal!");
    }
    for (int i = 0; i < t1->ndim; i++)
    {
        if (t1->shape[i] != t2->shape[i])
        {
            perror("Shapes are not equal!");
        }
    }
}

tensor *t_alloc(int ndim, const int shape[ndim])
{
    tensor *t = malloc(sizeof(tensor));
    t->shape = malloc(sizeof(int) * ndim);
    t->_v_size = 1;
    for (int i = 0; i < ndim; i++)
    {
        t->_v_size *= shape[i];
        t->shape[i] = shape[i];
    }
    t->ndim = ndim;
    t->v = malloc(sizeof(param_t) * t->_v_size);
    return t;
}

void t_init_rand(tensor *t)
{
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = prand();
    }
}

tensor *t_copy(tensor *t)
{
    tensor *c = t_alloc(t->ndim, t->shape);
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = c->v[i];
    }
    return c;
}

void t_free(tensor *t)
{
    if (t->shape != NULL)
    {
        free(t->shape);
        t->shape = NULL;
    }
    if (t->v != NULL)
    {
        free(t->v);
        t->v = NULL;
    }
    free(t);
}

tensor *t_from_1dim_array(int d1_size, param_t array[d1_size])
{
    const int shape[1] = {d1_size};
    tensor *t = t_alloc(1, shape);
    for (int i = 0; i < d1_size; i++)
    {
        t->v[i] = array[i];
    }
    return t;
}
tensor *t_from_2dim_array(int d1_size, int d2_size, param_t array[d1_size][d2_size])
{
    const int shape[2] = {d1_size, d2_size};
    tensor *t = t_alloc(2, shape);
    for (int i = 0; i < d1_size; i++)
    {
        for (int j = 0; j < d2_size; j++)
        {
            int idx = i * d2_size + j;
            t->v[idx] = array[i][j];
        }
    }
    return t;
}

tensor *t_from_3dim_array(int d1_size, int d2_size, int d3_size, param_t array[d1_size][d2_size][d3_size])
{
    const int shape[3] = {d1_size, d2_size, d3_size};
    tensor *t = t_alloc(3, shape);
    for (int i = 0; i < d1_size; i++)
    {
        for (int j = 0; j < d2_size; j++)
        {
            for (int k = 0; k < d3_size; k++)
            {
                int idx = i * d2_size + j * d3_size + k;
                t->v[idx] = array[i][j][k];
            }
        }
    }
    return t;
}

/**
 * Operations
 * All operations are in-place by default. make a copy before if you want to
 * keep the destination in-tact. They do return the destination again however,
 * making it for easier chaining.
 */
tensor *t_elem_add(tensor *dst, tensor *add)
{
    _assert_shape_equals(dst, add);
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] += add->v[i];
    }
    return dst;
}

tensor *t_elem_sub(tensor *dst, tensor *add)
{
    _assert_shape_equals(dst, add);
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] -= add->v[i];
    }
    return dst;
}

tensor *t_elem_mul(tensor *dst, tensor *add)
{
    _assert_shape_equals(dst, add);
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] *= add->v[i];
    }
    return dst;
}

tensor *t_elem_div(tensor *dst, tensor *add)
{
    _assert_shape_equals(dst, add);
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] /= add->v[i];
    }
    return dst;
}

tensor *t_add_const(tensor *dst, param_t cnst)
{
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] += cnst;
    }
    return dst;
}

tensor *t_sub_const(tensor *dst, param_t cnst)
{
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] -= cnst;
    }
    return dst;
}

tensor *t_mul_const(tensor *dst, param_t cnst)
{
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] *= cnst;
    }
    return dst;
}

tensor *t_div_const(tensor *dst, param_t cnst)
{
    // faster if we just iterate over _v_size than the dimensions
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] /= cnst;
    }
    return dst;
}

// Reducing actions
tensor *t_collapse_sum(tensor *t)
{
    // Collapses along the last axis
    // Calculate shape of new tensor
    // for (int d = 0; d < t->ndim; d++)
    // {
    // }
}

/**
 * Utils
 */
/*
Returns a dimension-referencing integer (bitwise) to target certain dimensions
e.g. the third dimension equals the third bit to be '1' aka 0...100 (=8)
See also the T_DIM_x constants
*/
int t_dim(int dim)
{
    return 1 << (dim - 1);
}

void t_print(tensor *t)
{
    int *dims_ptr = calloc(t->ndim, sizeof(int));
    for (int i = 0; i < t->_v_size; i++)
    {
        int rem = i;
        for (int d = 0; d < t->ndim; d++)
        {
            int d_idx = -1;
            if (d < t->ndim - 1)
            {
                d_idx = rem / t->shape[d + 1];
                rem -= d_idx * t->shape[d + 1];
                if (d_idx > 0)
                {
                    if (dims_ptr[d] != d_idx)
                        printf("]\n");
                }
                else
                {
                    if (dims_ptr[d] != d_idx || i == 0)
                        printf("[");
                }
            }
            else
            {
                d_idx = rem;
                if (d_idx > 0)
                    printf(", ");
                else
                    printf("[");
            }
            dims_ptr[d] = d_idx;
        }
        printf("%.4f", t->v[i]);
    }
    for (int d = 0; d < t->ndim; d++)
    {
        printf("]");
    }
}