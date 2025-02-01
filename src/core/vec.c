#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "core.h"

vec vec_alloc(int n)
{
    return malloc(sizeof(param_t) * n);
}
void vec_free(vec v)
{
    free(v);
}

vec vec_alloc_rand(int n)
{
    vec v = malloc(sizeof(param_t) * n);
    for (int i = 0; i < n; i++)
    {
        v[i] = prand();
    }
    return v;
}
mat vec_transposed_vec_mul(vec a, vec b, param_t const_factor, int sa, int sb)
{
    mat m = mat_alloc(sa, sb);
    for (int ia = 0; ia < sa; ia++)
    {
        for (int ib = 0; ib < sb; ib++)
        {
            m[ia][ib] = const_factor * a[ia] * b[ib];
        }
    }
    return m;
}
vec vec_elem_mul(vec a, vec b, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = a[i] * b[i];
    }
    return r;
}

vec vec_elem_add(vec a, vec b, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = a[i] + b[i];
    }
    return r;
}

void vec_norm(vec v, int n)
{
    param_t vec_len = 0;
    for (int i = 0; i < n; i++)
    {
        vec_len += v[i] * v[i];
    }
    vec_len = sqrt(vec_len);
    for (int i = 0; i < n; i++)
    {
        v[i] = v[i] / vec_len;
    }
}
vec vec_elem_sub_mul(vec a, vec b, param_t subFactor, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = a[i] - subFactor * b[i];
    }
    return r;
}

vec vec_elem_add_mul(vec a, vec b, param_t factor, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = a[i] + factor * b[i];
    }
    return r;
}

vec vec_elem_sub(vec a, vec b, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = a[i] - b[i];
    }
    return r;
}

param_t vec_collapse_sum(vec v, int n)
{
    param_t sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += v[i];
    }
    return sum;
}

vec vec_add_const(vec v, param_t c, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = v[i] + c;
    }
    return r;
}

void vec_apply_ip(vec v, param_t (*applyFn)(param_t), int n)
{
    for (int i = 0; i < n; i++)
    {
        v[i] = applyFn(v[i]);
    }
}

vec vec_mul_const(vec v, param_t c, int n)
{
    vec r = vec_alloc(n);
    for (int i = 0; i < n; i++)
    {
        r[i] = v[i] * c;
    }
    return r;
}

vec vec_from_mat_col(int colIdx, int rows, mat m)
{
    return vec_from_array(rows, m[colIdx]);
}

vec vec_from_array(int rows, param_t a[rows])
{
    vec v = vec_alloc(rows);
    for (int i = 0; i < rows; i++)
    {
        v[i] = a[i];
    }
    return v;
}

vec *vec_array_from_array_of_arrays(int numVecs, int numElems, param_t a[numVecs][numElems])
{
    vec *vecs = malloc(numVecs * sizeof(vec *));
    for (int i = 0; i < numVecs; i++)
    {
        vecs[i] = vec_from_array(numElems, a[i]);
    }
    return vecs;
}

vec vec_from_single(param_t val)
{
    vec v = vec_alloc(1);
    v[0] = val;
    return v;
}

void vec_print(vec v, int n)
{
    printf("[");
    for (int i = 0; i < n; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%f", v[i]);
    }
    printf("]");
}