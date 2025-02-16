#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "core.h"

#define TEST_FAILURE 0
#define TEST_SUCCESS 1

// assertions
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

int __vassert_tensor_equals(tensor *actual, tensor *expected)
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

int __vassert_int_array_equals(int *actual, int *expected, int size)
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

int __vassert_int_equals(int actual, int expected)
{
    if (actual != expected)
    {
        printf("Integers are not equal: %i != %i", actual, expected);
        return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}

// Test cases
int test_from_1dim_array()
{
    // arrange
    const int SIZE = 5;
    param_t in_and_out[SIZE] = {5, 3, 3, 2, 1};

    // act
    tensor *res = t_from_1dim_array(SIZE, in_and_out);

    // assert
    for (int i = 0; i < SIZE; i++)
    {
        if (res->v[i] != in_and_out[i])
            return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}

int test_from_2dim_array()
{
    // arrange
    const int SIZE_X = 5;
    const int SIZE_Y = 2;
    param_t in_and_out[SIZE_X][SIZE_Y] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};

    // act
    tensor *res = t_from_2dim_array(SIZE_X, SIZE_Y, in_and_out);

    // assert
    for (int x = 0; x < SIZE_X; x++)
    {
        for (int y = 0; y < SIZE_Y; y++)
        {
            if (res->v[x * SIZE_Y + y] != in_and_out[x][y])
            {
                t_print(res);
                return TEST_FAILURE;
            }
        }
    }
    return TEST_SUCCESS;
}

int test_from_3dim_array()
{
    // arrange
    const int SIZE_X = 4;
    const int SIZE_Y = 3;
    const int SIZE_Z = 2;
    param_t in_and_out[SIZE_X][SIZE_Y][SIZE_Z] = {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}, {{13, 14}, {15, 16}, {17, 18}}, {{19, 20}, {21, 22}, {23, 24}}};

    // act
    tensor *res = t_from_3dim_array(SIZE_X, SIZE_Y, SIZE_Z, in_and_out);

    // assert
    for (int x = 0; x < SIZE_X; x++)
    {
        for (int y = 0; y < SIZE_Y; y++)
        {
            for (int z = 0; z < SIZE_Z; z++)
            {
                if (res->v[x * SIZE_Y * SIZE_Z + y * SIZE_Z + z] != in_and_out[x][y][z])
                {
                    t_print(res);
                    return TEST_FAILURE;
                }
            }
        }
    }
    return TEST_SUCCESS;
}

int test_append_dim()
{
    // arrange
    param_t vecIn[5] = {5, 4, 3, 2, 1};
    param_t matOut[5][1] = {{5}, {4}, {3}, {2}, {1}};

    tensor *in = t_from_1dim_array(5, vecIn);
    tensor *out = t_from_2dim_array(5, 1, matOut);

    // act
    tensor *act = t_append_dim(in);

    // assert
    return __vassert_tensor_equals(act, out);
}

int test_flatten_dims()
{
    // arrange
    param_t matIn[5][1] = {{5}, {4}, {3}, {2}, {1}};
    param_t vecOut[5] = {5, 4, 3, 2, 1};

    tensor *in = t_from_2dim_array(5, 1, matIn);
    tensor *out = t_from_1dim_array(5, vecOut);

    // act
    tensor *act = t_flatten_dims(in);

    // assert
    return __vassert_tensor_equals(act, out);
}

int test_vector_transpose()
{
    // arrange
    param_t vecIn[5] = {5, 4, 3, 2, 1};
    param_t matOut[1][5] = {{5, 4, 3, 2, 1}};
    tensor *in = t_from_1dim_array(5, vecIn);
    tensor *out = t_from_2dim_array(1, 5, matOut);

    // act
    tensor *res = t_transpose(in, 1);

    // assert
    return __vassert_tensor_equals(res, out);
}

int test_mat_transpose()
{
    // arrange
    param_t matIn[3][3] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    param_t matOut[3][3] = {{9, 6, 3}, {8, 5, 2}, {7, 4, 1}};
    tensor *in = t_from_2dim_array(3, 3, matIn);
    tensor *out = t_from_2dim_array(3, 3, matOut);

    // act
    tensor *res = t_transpose(in, 2);

    // assert
    return __vassert_tensor_equals(res, out);
}

int test_mul_vec_vec_transpose()
{
    // arrange
    param_t vecA[4] = {1, 2, 3, 4};
    param_t vecB[4] = {5, 6, 7, 8};
    param_t matExp[4][4] = {{1 * 5, 1 * 6, 1 * 7, 1 * 8}, {2 * 5, 2 * 6, 2 * 7, 2 * 8}, {3 * 5, 3 * 6, 3 * 7, 3 * 8}, {4 * 5, 4 * 6, 4 * 7, 4 * 8}};

    tensor *tA = t_from_1dim_array(4, vecA);
    tensor *tB = t_from_1dim_array(4, vecB);
    tensor *tExp = t_from_2dim_array(4, 4, matExp);

    // act
    tensor *tAct = t_mul(tA, t_transpose(tB, 1));

    // assert
    return __vassert_tensor_equals(tAct, tExp);
}

int test_mul_vec_tranpose_vec_fixed()
{
    // arrange
    param_t vecA[1][4] = {{1, 2, 3, 4}};
    param_t vecB[4][1] = {{5}, {6}, {7}, {8}};
    param_t scaExp = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8;

    tensor *tA = t_from_2dim_array(1, 4, vecA);
    tensor *tB = t_from_2dim_array(4, 1, vecB);
    tensor *tExp = t_alloc_single_from(scaExp);

    // act
    tensor *tAct = t_mul(tA, tB);

    // assert
    return __vassert_tensor_equals(tAct, tExp);
}

int test_mul_vec_tranpose_vec()
{
    // arrange
    param_t vecA[4] = {1, 2, 3, 4};
    param_t vecB[4] = {5, 6, 7, 8};
    param_t scaExp = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8;

    tensor *tA = t_from_1dim_array(4, vecA);
    tensor *tB = t_from_1dim_array(4, vecB);
    tensor *tExp = t_alloc_single_from(scaExp);

    // act
    tensor *tAct = t_mul(t_transpose(tA, 1), tB);

    // assert
    return __vassert_tensor_equals(tAct, tExp);
}

int test_mul_mat_vec()
{
    // arrange
    param_t matA[2][3] = {{1, -1, 2}, {0, -3, 1}};
    param_t vecB[4] = {2, 1, 0};
    param_t vecExp[2] = {1, -3};

    tensor *tA = t_from_2dim_array(2, 3, matA);
    tensor *tB = t_from_1dim_array(3, vecB);
    tensor *tExp = t_from_1dim_array(2, vecExp);

    // act
    tensor *tAct = t_mul(tA, tB);

    // assert
    return __vassert_tensor_equals(tAct, tExp);
}

int test_mul_mat_mat()
{
    // arrange
    param_t matA[2][3] = {{0, 4, -2}, {-4, -3, 0}};
    param_t matB[3][2] = {{0, 1}, {1, -1}, {2, 3}};
    param_t matExp[2][2] = {{0, -10}, {-3, -1}};

    tensor *tA = t_from_2dim_array(2, 3, matA);
    tensor *tB = t_from_2dim_array(3, 2, matB);
    tensor *tExp = t_from_2dim_array(2, 2, matExp);

    // act
    tensor *tAct = t_mul(tA, tB);

    // assert
    return __vassert_tensor_equals(tAct, tExp);
}

int test_calc_strides()
{
    // arrange
    int shape[3] = {3, 4, 5};
    int expectedStrides[3] = {20, 5, 1};
    int outStrides[3];

    tensor *t_in = t_alloc(3, shape);
    // act
    t_calc_strides(t_in, outStrides);

    // assert
    return __vassert_int_array_equals(outStrides, expectedStrides, 3);
}

int test_get_flat_index()
{
    // arrange
    int shape[3] = {3, 4, 5};
    int strides[3];
    int indices[3] = {2, 3, 2};

    tensor *t_in = t_alloc(3, shape);
    t_calc_strides(t_in, strides);

    // act
    int res = t_get_flat_index(t_in, strides, indices);

    // assert
    return __vassert_int_equals(res, 57);
}

int test_get_indices()
{
    // arrange
    int shape[3] = {3, 4, 5};
    int strides[3], outIndices[3];
    int expectedIndices[3] = {2, 3, 2};

    tensor *t_in = t_alloc(3, shape);
    t_calc_strides(t_in, strides);

    // act
    t_get_indices(t_in, 57, strides, outIndices);

    // assert
    return __vassert_int_array_equals(outIndices, expectedIndices, 3);
}

// Utils
void *__test_thread(void *testFn)
{
    int *ret = malloc(sizeof(int));
    ret[0] = ((int (*)(void))testFn)();
    pthread_exit(ret);
}

void __run_test(const char *testName, int (*testFn)(void))
{
    pthread_t testThread;
    void *ret;
    printf("Running test case '%s' ..... ", testName);

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

void tensor_test()
{
    __run_test("test_from_1dim_array", test_from_1dim_array);
    __run_test("test_from_2dim_array", test_from_2dim_array);
    __run_test("test_from_3dim_array", test_from_2dim_array);
    __run_test("test_vector_transpose", test_vector_transpose);
    __run_test("test_mat_transpose", test_mat_transpose);
    __run_test("test_calc_strides", test_calc_strides);
    __run_test("test_get_flat_index", test_get_flat_index);
    __run_test("test_get_indices", test_get_indices);
    __run_test("test_mul_vec_vec_transpose", test_mul_vec_vec_transpose);
    __run_test("test_mul_vec_tranpose_vec", test_mul_vec_tranpose_vec);
    __run_test("test_mul_vec_tranpose_vec_fixed", test_mul_vec_tranpose_vec_fixed);
    __run_test("test_mul_mat_vec", test_mul_mat_vec);
    __run_test("test_mul_mat_mat", test_mul_mat_mat);
    __run_test("test_append_dim", test_append_dim);
    __run_test("test_flatten_dims", test_flatten_dims);
}