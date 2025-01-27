#include "tnet.h"

typedef param_t **mat;
typedef param_t *vec;

// Matrix
mat mat_alloc(int rows, int cols);
void mat_free(mat m, int rows);
vec mat_dot_product(mat m, vec v, int rows, int cols);
mat mat_from_array(int rows, int cols, param_t a[rows][cols]);

// Vector
vec vec_alloc(int n);
void vec_free(vec v);
vec vec_alloc_rand(int n);
vec vec_elem_mul(vec a, vec b, int n);
vec vec_elem_add(vec a, vec b, int n);
param_t vec_collapse_sum(vec v, int n);
vec vec_add_const(vec v, param_t c, int n);
vec vec_mul_const(vec v, param_t c, int n);
vec vec_from_array(int rows, param_t a[rows]);
vec vec_from_mat_col(int colIdx, int rows, mat m);
void vec_print(vec v, int n);

// Vector - In-Place
void vec_apply_ip(vec v, param_t (*applyFn)(param_t), int n);