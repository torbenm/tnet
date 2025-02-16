/**
 * Collection of test cases among the project.
 *
 */
#include "core.h"

#define TEST_FAILURE 0
#define TEST_SUCCESS 1

// Assertions - to be found in ./src/test.c
int assert_tensor_equals(tensor *actual, tensor *expected);
int assert_int_array_equals(int *actual, int *expected, int size);
int assert_int_equals(int actual, int expected);

// tensor_test.c
int test_from_1dim_array();
int test_from_2dim_array();
int test_from_3dim_array();
int test_append_dim();
int test_flatten_dims();
int test_vector_transpose();
int test_mat_transpose();
int test_mul_vec_vec_transpose();
int test_mul_vec_tranpose_vec_fixed();
int test_mul_vec_tranpose_vec();
int test_mul_mat_vec();
int test_mul_mat_mat();
int test_calc_strides();
int test_get_flat_index();
int test_get_indices();

// Test Runner
typedef struct
{
    const char *name;
    int (*ptr)(void);
} TestMapEntry;

void run_test(const char *testName, int (*testFn)(void));
