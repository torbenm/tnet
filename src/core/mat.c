#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "core.h"

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

mat mat_alloc_rand(int rows, int cols)
{
    mat m = malloc(rows * sizeof(param_t *));
    for (int r = 0; r < rows; r++)
    {
        m[r] = vec_alloc_rand(cols);
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
mat mat_elem_add(mat m1, mat m2, int rows, int cols)
{
    mat o = mat_alloc(rows, cols);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            o[r][c] = m1[r][c] + m2[r][c];
        }
    }
    return o;
}
mat mat_transpose(mat m, int rows, int cols)
{
    mat t = mat_alloc(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            t[j][i] = m[i][j];
        }
    }
    return t;
}
void mat_print(mat m, int rows, int cols)
{
    printf("[");
    for (int i = 0; i < rows; i++)
    {
        if (i > 0)
            printf(", ");
        vec_print(m[i], cols);
    }
    printf("]");
}
void mat_norm(mat m, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        vec_norm(m[i], cols);
    }
}
void mat_free(mat m, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(m[i]);
    }
    free(m);
}
