#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "core.h"
#include "test.h"

int main()
{
    TestMapEntry tests[] = {
        // tensor_test.c
        {"test_from_1dim_array", test_from_1dim_array},
        {"test_from_2dim_array", test_from_2dim_array},
        {"test_from_3dim_array", test_from_2dim_array},
        {"test_vector_transpose", test_vector_transpose},
        {"test_mat_transpose", test_mat_transpose},
        {"test_calc_strides", test_calc_strides},
        {"test_get_flat_index", test_get_flat_index},
        {"test_get_indices", test_get_indices},
        {"test_mul_vec_vec_transpose", test_mul_vec_vec_transpose},
        {"test_mul_vec_tranpose_vec", test_mul_vec_tranpose_vec},
        {"test_mul_vec_tranpose_vec_fixed", test_mul_vec_tranpose_vec_fixed},
        {"test_mul_mat_vec", test_mul_mat_vec},
        {"test_mul_mat_mat", test_mul_mat_mat},
        {"test_append_dim", test_append_dim},
        {"test_flatten_dims", test_flatten_dims},
        // ...
        {NULL, NULL} // sentinel pointer
    };

    print_header("Running tests");
    for (int i = 0; tests[i].ptr != NULL; i++)
    {
        run_test(tests[i].name, tests[i].ptr);
    }
    return 0; // success regardless of what happens
}

void *__test_thread(void *testFn)
{
    int *ret = malloc(sizeof(int));
    ret[0] = ((int (*)(void))testFn)();
    pthread_exit(ret);
}

void run_test(const char *testName, int (*testFn)(void))
{
    pthread_t testThread;
    void *ret;
    printf("%-50s", testName);

    if (pthread_create(&testThread, NULL, __test_thread, testFn) != 0)
    {
        error("failed to start thread.\n");
    }

    if (pthread_join(testThread, &ret) != 0)
    {
        error("failed to join thread.\n");
    }
    if (ret != NULL && ((int *)ret)[0] == TEST_SUCCESS)
        printf(" ✅");
    else
        printf(" ❌");
    printf("\n");
    free(ret);
}

/**
 * Assertion functions
 */

int __assert_tensor_equals(tensor *actual, tensor *expected)
{
    if (actual->ndim != expected->ndim)
        return TEST_FAILURE;
    for (int i = 0; i < actual->ndim; i++)
        if (actual->shape[i] != expected->shape[i])
            return TEST_FAILURE;
    for (int i = 0; i < actual->_v_size; i++)
    {
        if (actual->v[i] != expected->v[i])
            return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}

int assert_tensor_equals(tensor *actual, tensor *expected)
{
    // same as above, but printing if tensors are not equal
    if (__assert_tensor_equals(actual, expected) == TEST_FAILURE)
    {
        printf("The following tensors are not equal.\n");
        t_print(actual);
        printf("\n!=\n");
        t_print(expected);
        return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}

int assert_int_array_equals(int *actual, int *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (actual[i] != expected[i])
        {
            printf("Arrays are not equal: ");
            print_int_array(actual, size);
            printf(" != ");
            print_int_array(expected, size);
            return TEST_FAILURE;
        }
    }
    return TEST_SUCCESS;
}

int assert_int_equals(int actual, int expected)
{
    if (actual != expected)
    {
        printf("Integers are not equal: %i != %i", actual, expected);
        return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}