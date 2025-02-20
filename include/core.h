#pragma once

#include <stdlib.h>

// Base Types
typedef double param_t;
typedef param_t **mat;
typedef param_t *vec;

#define EPSILON 10e-8

/**
 * Tensor definitions.
 * Tensors are multidimensional matrices.
 * We store them in a single flat array with a fixed shape.
 */

// Types
typedef struct tensor
{
    int locked; // whether or not the tensor might be updated in-place
    int ndim;
    int *shape;
    int _v_size; // size of the v array
    param_t *v;
} tensor;

// alloc & init & locking
tensor *t_alloc(int ndim, const int shape[ndim]);
tensor *t_null();
void t_mark(tensor *t);
tensor *t_alloc_single();
tensor *t_alloc_single_from(param_t value);
tensor *t_alloc_rand(int ndim, const int shape[ndim]);
tensor *t_copy(tensor *t);
tensor *t_copy_or_add(tensor **dst, tensor *src);
void t_free(tensor *t);
void t_init_rand(tensor *t);
void t_init_xavier(tensor *t);
void t_init_const(tensor *t, const param_t cnst);
tensor *t_lock(tensor *t);
void t_assert_not_locked(tensor *t);
tensor *t_identity(int size);
tensor *t_diag(tensor *vec);

// Utils
tensor *t_from_1dim_array(int d1_size, param_t array[d1_size]);
tensor *t_from_2dim_array(int d1_size, int d2_size, param_t array[d1_size][d2_size]);
tensor *t_from_3dim_array(int d1_size, int d2_size, int d3_size, param_t array[d1_size][d2_size][d3_size]);
int t_is_single_element(tensor *t);
void t_print(tensor *t);
tensor *t_append_dim(tensor *t);
tensor *t_prepend_dim(tensor *t);
tensor *t_flatten_dims(tensor *t);
void t_print_shape(tensor *t);
void t_calc_strides(tensor *t, int *outStrides);
int t_get_flat_index(tensor *t, int *strides, int *indices);
void t_get_indices(tensor *t, int flatIndex, int *strides, int *outIndices);
tensor *t_flatten(tensor *dst);

// Element wise operations (tensor x tensor and tensor x scalar)
tensor *t_elem_add(tensor *dst, tensor *add);
tensor *t_elem_sub(tensor *dst, tensor *add);
tensor *t_elem_mul(tensor *dst, tensor *add);
tensor *t_elem_div(tensor *dst, tensor *add);
tensor *t_add_const(tensor *dst, param_t cnst);
tensor *t_sub_const(tensor *dst, param_t cnst);
tensor *t_mul_const(tensor *dst, param_t cnst);
tensor *t_div_const(tensor *dst, param_t cnst);
tensor *t_pow_const(tensor *dst, param_t cnst);
tensor *t_apply(tensor *dst, param_t (*applyFn)(param_t));

tensor *t_transpose(tensor *t, int transposeLastNDimensions);

// Tensor multiplication
tensor *t_mul(tensor *a, tensor *b);

// Tensor - Collapsing
tensor *t_collapse_sum(tensor *t, int dimIdx);
param_t t_collapse_sum_all(tensor *t);
param_t t_collapse_mean_all(tensor *t);

// tnet init
void tnet_init();

// Utils
param_t prand();
void error(const char *msg, ...);
void print_int_array(int *a, int size);
void print_header(const char *msg, ...);
param_t clip(param_t, param_t, param_t);

// testing...
void tensor_test();

// Memory Mgmt
void *mm_alloc(size_t size);
void *mm_calloc(size_t __count, size_t __size);
void mm_mark(void *ptr);
void mm_unmark(void *ptr);
void mm_unmark_all();
void mm_sweep();
void mm_free(void *ptr);