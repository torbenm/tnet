#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int is_single_tensor(tensor *t)
{
    return t->ndim == 1 && t->shape[0] == 1;
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
    t->v = calloc(t->_v_size, sizeof(param_t));
    return t;
}

tensor *t_alloc_single()
{
    int shape[1] = {1};
    return t_alloc(1, shape);
}

void t_init_rand(tensor *t)
{
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = prand();
    }
}

void t_init_const(tensor *t, const param_t cnst)
{
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = cnst;
    }
}

tensor *t_copy(tensor *t)
{
    tensor *c = t_alloc(t->ndim, t->shape);
    for (int i = 0; i < c->_v_size; i++)
    {
        c->v[i] = t->v[i];
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
param_t _add(param_t a, param_t b) { return a + b; }
param_t _sub(param_t a, param_t b) { return a - b; }
param_t _mul(param_t a, param_t b) { return a * b; }
param_t _div(param_t a, param_t b) { return a / b; }

tensor *_t_const_apply(tensor *dst, param_t cnst, param_t (*fn)(param_t, param_t))
{
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] = fn(dst->v[i], cnst);
    }
    return dst;
}

tensor *_t_elem_apply(tensor *dst, tensor *snd, param_t (*fn)(param_t, param_t))
{
    if (is_single_tensor(snd)) // if snd is single tensor, we apply the value to all equally (treat it as a scalar)
        return _t_const_apply(dst, snd->v[0], fn);

    _assert_shape_equals(dst, snd);

    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] = fn(dst->v[i], snd->v[i]);
    }
    return dst;
}

tensor *t_elem_add(tensor *dst, tensor *add)
{
    return _t_elem_apply(dst, add, _add);
}

tensor *t_elem_sub(tensor *dst, tensor *add)
{
    return _t_elem_apply(dst, add, _sub);
}

tensor *t_elem_mul(tensor *dst, tensor *add)
{
    return _t_elem_apply(dst, add, _mul);
}

tensor *t_elem_div(tensor *dst, tensor *add)
{
    return _t_elem_apply(dst, add, _div);
}

tensor *t_add_const(tensor *dst, param_t cnst)
{
    return _t_const_apply(dst, cnst, _add);
}

tensor *t_sub_const(tensor *dst, param_t cnst)
{
    return _t_const_apply(dst, cnst, _sub);
}

tensor *t_mul_const(tensor *dst, param_t cnst)
{
    return _t_const_apply(dst, cnst, _mul);
}

tensor *t_div_const(tensor *dst, param_t cnst)
{
    return _t_const_apply(dst, cnst, _div);
}

tensor *t_apply(tensor *dst, param_t (*applyFn)(param_t))
{
    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] = applyFn(dst->v[i]);
    }
    return dst;
}

// Reducing actions
/**
 * Collapses tensor across the given dim.
 * Either the dim index (0, 1, ...) or -1 for the last dim.
 */
tensor *t_collapse_sum(tensor *t, int collapseDimIdx)
{
    if (t->ndim == 1)
    {
        // special case: we only have one dimension, simply adding all up
        tensor *collapsed = t_alloc_single();
        for (int i = 0; i < t->_v_size; i++)
            collapsed->v[0] += t->v[i];
        return collapsed;
    }

    if (collapseDimIdx < 0)
        collapseDimIdx = t->ndim - 1;
    else if (collapseDimIdx >= t->ndim)
        perror("Trying to collapse across non-existing dimension.");

    int collapsedShape[t->ndim - 1];
    int collapsedShapeIdx = 0;
    for (int d = 0; d < t->ndim; d++)
    {
        if (d != collapseDimIdx)
            collapsedShape[collapsedShapeIdx++] = t->shape[d];
    }

    tensor *collapsed = t_alloc(t->ndim - 1, collapsedShape);
    for (int i = 0; i < t->_v_size; i++)
    {
        int collapsedIdx = 0;
        int collapsedDimIdx = 0;
        int rem = i;

        // Calculate the index in the collapsed array
        // by first calculating the dimension indices
        // and then translating them to the new one
        for (int d = 0; d < t->ndim; d++)
        {
            int d_idx = -1;
            // Calculate dimension indicies of the old matrix
            if (d < t->ndim - 1)
            {
                d_idx = rem / t->shape[d + 1];
                rem -= d_idx * t->shape[d + 1];
            }
            else
            {
                d_idx = rem;
            }

            // calculate index of the new matrix & skipping dimIdx
            if (d != collapseDimIdx)
            {
                if (collapsedDimIdx < collapsed->ndim - 1)
                {
                    collapsedIdx += collapsed->shape[collapsedDimIdx + 1] * d_idx;
                }
                else
                {
                    collapsedIdx += d_idx;
                }
                collapsedDimIdx++;
            }
        }
        collapsed->v[collapsedIdx] += t->v[i]; // adding since we are summing
    }

    return collapsed;
}

/**
 * Utils
 */
void t_print(tensor *t)
{
    if (t->ndim == 1 && t->shape[0] == 1)
    {
        // single element tensor
        printf("%.4f", t->v[0]);
        return;
    }

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