#pragma once

// Base Types
typedef double param_t;
typedef param_t **mat;
typedef param_t *vec;

// Matrix
mat mat_alloc(int rows, int cols);
mat mat_alloc_rand(int rows, int cols);
void mat_free(mat m, int rows);
mat mat_transpose(mat m, int rows, int cols);
vec mat_dot_product(mat m, vec v, int rows, int cols);
mat mat_mul_const(mat m, param_t factor, int rows, int cols);
mat mat_elem_add(mat m1, mat m2, int rows, int cols);
mat mat_elem_sub(mat m1, mat m2, int rows, int cols);
mat mat_elem_sub_mul(mat m1, mat m2, param_t subFactor, int rows, int cols);
mat mat_elem_add_mul(mat m1, mat m2, param_t factor, int rows, int cols);
mat mat_from_array(int rows, int cols, param_t a[rows][cols]);
mat mat_from_vec(vec v);
mat mat_from_singledim_array(int elems, param_t a[elems]);
void mat_norm(mat m, int rows, int cols);
void mat_print(mat m, int rows, int cols);

// Vector
vec vec_alloc(int n);
void vec_free(vec v);
vec vec_alloc_rand(int n);
vec vec_elem_mul(vec a, vec b, int n);
vec vec_elem_add(vec a, vec b, int n);
vec vec_elem_sub(vec a, vec b, int n);
vec vec_elem_sub_mul(vec a, vec b, param_t subFactor, int n);
vec vec_elem_add_mul(vec a, vec b, param_t factor, int n);
void vec_norm(vec v, int n);
param_t vec_collapse_sum(vec v, int n);
vec vec_add_const(vec v, param_t c, int n);
vec vec_mul_const(vec v, param_t c, int n);
mat vec_transposed_vec_mul(vec a, vec b, param_t const_factor, int sa, int sb);
vec vec_from_array(int rows, param_t a[rows]);
vec *vec_array_from_array_of_arrays(int numVecs, int numElems, param_t a[numVecs][numElems]);
vec vec_from_mat_col(int colIdx, int rows, mat m);
vec vec_from_single(param_t val);
void vec_print(vec v, int n);

// Vector - In-Place
void vec_apply_ip(vec v, param_t (*applyFn)(param_t), int n);

/**
 * Tensor definitions.
 * Tensors are multidimensional matrices.
 * We store them in a single flat array with a fixed shape.
 */

// Types
typedef struct tensor
{
    int ndim;
    int *shape;
    int _v_size; // size of the v array
    param_t *v;
} tensor;

// alloc & init
tensor *t_alloc(int ndim, const int shape[ndim]);
tensor *t_alloc_single();
tensor *t_copy(tensor *t);
void t_free(tensor *t);
void t_init_rand(tensor *t);
void t_init_const(tensor *t, const param_t cnst);

// Utils
tensor *t_from_1dim_array(int d1_size, param_t array[d1_size]);
tensor *t_from_2dim_array(int d1_size, int d2_size, param_t array[d1_size][d2_size]);
tensor *t_from_3dim_array(int d1_size, int d2_size, int d3_size, param_t array[d1_size][d2_size][d3_size]);
int is_single_tensor(tensor *t);
void t_print(tensor *t);

// Element wise operations (tensor x tensor and tensor x scalar)
tensor *t_elem_add(tensor *dst, tensor *add);
tensor *t_elem_sub(tensor *dst, tensor *add);
tensor *t_elem_mul(tensor *dst, tensor *add);
tensor *t_elem_div(tensor *dst, tensor *add);
tensor *t_add_const(tensor *dst, param_t cnst);
tensor *t_sub_const(tensor *dst, param_t cnst);
tensor *t_mul_const(tensor *dst, param_t cnst);
tensor *t_div_const(tensor *dst, param_t cnst);
tensor *t_apply(tensor *dst, param_t (*applyFn)(param_t));

// Tensor - Collapsing
tensor *t_collapse_sum(tensor *t, int dimIdx);

// tnet init
void tnet_init();

// Utils
param_t prand();