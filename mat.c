#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tnet.h"
#include "mat.h"
#include "util.h"

// Matrix
mat mat_alloc(int rows, int cols)
{
    mat m = malloc(rows * sizeof(param_t *));
    for (int r = 0; r < rows; r++)
    {
        m[r] = malloc(cols * sizeof(param_t));
    }
    return m;
}
mat mat_from_array(int rows, int cols, param_t a[rows][cols])
{
    mat m = mat_alloc(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            m[i][j] = a[i][j];
        }
    }
    return m;
}
vec mat_dot_product(mat m, vec v, int rows, int cols)
{
    vec o = vec_alloc(rows);
    for (int r = 0; r < rows; r++)
    {
        o[r] = 0.0;
        for (int c = 0; c < cols; c++)
        {
            o[r] += m[r][c] * v[c];
        }
    }
    return o;
}
void mat_free(mat m, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(m[i]);
    }
    free(m);
}

// Vector
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