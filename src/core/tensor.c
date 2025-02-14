#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core.h"

/**
 * Private tensor helper functions
 */
void __assert_ndims(tensor *t, int ndim)
{
    if (t->ndim != ndim)
    {
        error("Shapes are not equal! Number of dimensions: %i != %i.", t->ndim, ndim);
    }
}
void __assert_shape(tensor *t, int ndim, int *shape)
{
    __assert_ndims(t, ndim);
    for (int i = 0; i < ndim; i++)
    {
        if (t->shape[i] != shape[i])
        {
            error("Shapes are not equal!");
        }
    }
}

void __assert_shape_equals(tensor *t1, tensor *t2)
{
    __assert_shape(t1, t2->ndim, t2->shape);
}

void t_assert_not_locked(tensor *t)
{
    if (t->locked)
        error("May not alter tensor in place - tensor is locked.");
}

// Helper functions for indices
void t_calc_strides(tensor *t, int *outStrides)
{
    // calculates the number of jumps we have to do in the flat index
    // to get to the position of a certain dimension.
    // e.g. for a [3,4,5] shape tensor we have strides [20, 5, 1]
    outStrides[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--)
    {
        outStrides[i] = t->shape[i + 1] * outStrides[i + 1];
    }
}

int t_get_flat_index(tensor *t, int *strides, int *indices)
{
    // e.g. for [3, 4, 5] tensor with indices [0, 1, 2]
    // we have strides [20, 5, 1] and want to have
    // flatIndex = 0 * 20 + 5 * 1 + 2 * 1 = 7
    // or for [2, 3, 2] = 40 + 15 + 2 = 57

    int flatIndex = 0;
    for (int i = 0; i < t->ndim; i++)
    {
        flatIndex += indices[i] * strides[i];
    }

    return flatIndex;
}

void t_get_indices(tensor *t, int flatIndex, int *strides, int *outIndices)
{
    // e.g. for [3, 4, 5] tensor with flatIndex 57
    // we have strides [20, 5, 1] and want to have
    // index = [2, 3, 2]
    int remainder = flatIndex;
    for (int d = 0; d < t->ndim - 1; d++)
    {
        outIndices[d] = remainder / strides[d];
        remainder = remainder % strides[d];
    }
    outIndices[t->ndim - 1] = remainder;
}

void __copy_array(int *inputArray, int *outArray, int lenToCopy)
{
    for (int i = 0; i < lenToCopy; i++)
        outArray[i] = inputArray[i];
}

void __t_copy_shape(tensor *t, int numDimsToCopy, int *outShape)
{
    if (t->ndim < numDimsToCopy)
        error("Number of dims to copy is larger than number of available dims");

    __copy_array(t->shape, outShape, numDimsToCopy);
}

tensor *t_add_dim(tensor *t)
{
    // adds a dimension which can be useful for
    // transforming vectors into fake matrices. e.g. [5] shape turns into [5, 1] shape.
    int shape[t->ndim + 1];
    __t_copy_shape(t, t->ndim, shape);
    shape[t->ndim] = 1;
    tensor *added = t_alloc(t->ndim + 1, shape);
    for (int i = 0; i < t->_v_size; i++)
    {
        added->v[i] = t->v[i];
    }
    return added;
}

/**
 * Allocation, Initialization & freeing methods
 */
tensor *t_alloc(int ndim, const int shape[ndim])
{
    tensor *t = mm_alloc(sizeof(tensor));
    t->shape = mm_alloc(sizeof(int) * ndim);
    t->_v_size = 1;
    t->locked = 0;
    for (int i = 0; i < ndim; i++)
    {
        t->_v_size *= shape[i];
        t->shape[i] = shape[i];
    }
    t->ndim = ndim;

    t->v = mm_calloc(t->_v_size, sizeof(param_t));
    return t;
}

void t_mark(tensor *t)
{
    if (t == NULL)
        return;
    mm_mark(t);
    mm_mark(t->v);
    mm_mark(t->shape);
}

tensor *t_alloc_single()
{
    int shape[1] = {1};
    return t_alloc(1, shape);
}

tensor *t_alloc_single_from(param_t value)
{
    int shape[1] = {1};
    tensor *t = t_alloc(1, shape);
    t->v[0] = value;
    return t;
}

tensor *t_alloc_rand(int ndim, const int shape[ndim])
{
    tensor *t = t_alloc(ndim, shape);
    t_init_rand(t);
    return t;
}

void t_init_rand(tensor *t)
{
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = prand();
    }
}

void t_init_xavier(tensor *t)
{
    int scale_sum = 0;
    for (int d = 0; d < t->ndim; d++)
        scale_sum += t->shape[d];
    param_t scale = sqrt(6.0 / (param_t)scale_sum);
    for (int i = 0; i < t->_v_size; i++)
    {
        t->v[i] = scale * prand();
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

tensor *t_copy_or_add(tensor **dst, tensor *src)
{
    if (*dst == NULL)
        *dst = t_copy(src);
    else
        t_elem_add(*dst, src);
    return *dst;
}

void t_free(tensor *t)
{
    if (t == NULL)
        return;
    if (t->shape != NULL)
    {
        mm_free(t->shape);
        t->shape = NULL;
    }
    if (t->v != NULL)
    {
        mm_free(t->v);
        t->v = NULL;
    }
    mm_free(t);
}

tensor *t_lock(tensor *t)
{
    if (t == NULL)
        return t;
    t->locked = 1;
    return t;
}

tensor *t_identity(int size)
{
    int shape[2] = {size, size};
    tensor *t = t_alloc(2, shape);
    for (int i = 0; i < size; i++)
    {
        t->v[i * size + i] = 1;
    }
    return t;
}

tensor *t_diag(tensor *vec)
{
    __assert_ndims(vec, 1);
    int shape[2] = {vec->_v_size, vec->_v_size};
    tensor *t = t_alloc(2, shape);
    for (int i = 0; i < vec->_v_size; i++)
    {
        t->v[i * vec->_v_size + i] = vec->v[i];
    }
    return t;
}

/**
 * Utility methods
 */
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
    int indices[2], strides[2];

    tensor *t = t_alloc(2, shape);

    t_calc_strides(t, strides);

    for (indices[0] = 0; indices[0] < shape[0]; indices[0]++)
    {

        for (indices[1] = 0; indices[1] < shape[1]; indices[1]++)
        {
            int flatIdx = t_get_flat_index(t, strides, indices);
            t->v[flatIdx] = array[indices[0]][indices[1]];
        }
    }
    return t;
}

tensor *t_from_3dim_array(int d1_size, int d2_size, int d3_size, param_t array[d1_size][d2_size][d3_size])
{
    const int shape[3] = {d1_size, d2_size, d3_size};
    int indices[3], strides[3];

    tensor *t = t_alloc(3, shape);

    t_calc_strides(t, strides);

    for (indices[0] = 0; indices[0] < shape[0]; indices[0]++)
    {
        for (indices[1] = 0; indices[1] < shape[1]; indices[1]++)
        {
            for (indices[2] = 0; indices[2] < shape[2]; indices[2]++)
            {
                int flatIdx = t_get_flat_index(t, strides, indices);
                t->v[flatIdx] = array[indices[0]][indices[1]][indices[2]];
            }
        }
    }
    return t;
}

int t_is_single_element(tensor *t)
{
    return t->ndim == 1 && t->shape[0] == 1;
}

tensor *t_flatten(tensor *dst)
{
    t_assert_not_locked(dst);
    int *newShape = mm_alloc(sizeof(int) * 1);
    newShape[0] = dst->_v_size;
    dst->ndim = 1;
    mm_free(dst->shape);
    dst->shape = newShape;
    return dst;
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
param_t _pow(param_t a, param_t b) { return pow(a, b); }

tensor *_t_const_apply(tensor *dst, param_t cnst, param_t (*fn)(param_t, param_t))
{
    t_assert_not_locked(dst);

    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] = fn(dst->v[i], cnst);
    }
    return dst;
}

tensor *_t_elem_apply(tensor *dst, tensor *snd, param_t (*fn)(param_t, param_t))
{
    if (t_is_single_element(snd)) // if snd is single tensor, we apply the value to all equally (treat it as a scalar)
        return _t_const_apply(dst, snd->v[0], fn);

    t_assert_not_locked(dst);
    __assert_shape_equals(dst, snd);

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

tensor *t_pow_const(tensor *dst, param_t cnst)
{
    return _t_const_apply(dst, cnst, _pow);
}

tensor *t_apply(tensor *dst, param_t (*applyFn)(param_t))
{
    t_assert_not_locked(dst);

    for (int i = 0; i < dst->_v_size; i++)
    {
        dst->v[i] = applyFn(dst->v[i]);
    }
    return dst;
}

// Tensor multiplication
tensor *t_mul(tensor *a, tensor *b)
{
    // This whole algorithm could be made easier
    if (b->ndim > 2)
        error("Second tensor may have most have 2 dimensions, found %i.\n", b->ndim);

    // get output shape. We are at most considering the last two dimensions
    // (m x n) * (n x p) = (m x p)

    tensor *originalA = a;
    tensor *originalB = b;
    if (a->ndim == 1)
        a = t_add_dim(a);

    if (b->ndim == 1)
        b = t_add_dim(b);

    int nA, m, nB, p, outputDims;
    nA = a->shape[a->ndim - 1];
    m = a->shape[a->ndim - 2];
    nB = b->shape[b->ndim - 2];
    p = b->shape[b->ndim - 1];
    int n = nA; // for ease of use

    if (n > 1)
        outputDims = originalA->ndim + originalB->ndim - 2;
    else
        outputDims = originalA->ndim + originalB->ndim - 1;

    if (nA != nB)
        error("Found non-matching dimensions for multiplication. m=%i, nA=%i, nB=%i, p=%i.\n", m, nA, nB, p);

    int outputShape[outputDims];
    // copy all shapes apart from the first one
    __t_copy_shape(a, a->ndim - 2, outputShape);
    if (p > 1)
    {
        outputShape[outputDims - 2] = m;
        outputShape[outputDims - 1] = p;
    }
    else
    {
        outputShape[outputDims - 1] = m;
        // last dimension with p = 1 is being ignored.
    }

    tensor *r = t_alloc(outputDims, outputShape);
    int indicesA[a->ndim], indicesB[b->ndim], indicesR[r->ndim], stridesA[a->ndim], stridesB[b->ndim], stridesR[r->ndim];

    t_calc_strides(a, stridesA);
    t_calc_strides(b, stridesB);
    t_calc_strides(r, stridesR);

    for (int flatIdx = 0; flatIdx < a->_v_size; flatIdx++)
    {
        t_get_indices(a, flatIdx, stridesA, indicesA);
        // formula is r_ik = sum(a_ij * b_jk);
        // with 1 <= i <= m; 1 <= k <= p; 1 <= j <= n(A/B)
        __copy_array(indicesA, indicesR, a->ndim - 2);

        int i, j, k;
        i = indicesA[a->ndim - 2];
        j = indicesA[a->ndim - 1];
        indicesB[b->ndim - 2] = j;
        for (int k = 0; k < p; k++)
        {
            if (p > 1)
            {
                indicesR[r->ndim - 2] = i;
                indicesR[r->ndim - 1] = k;
            }
            else
            {
                indicesR[r->ndim - 1] = i; // k will always 0
            }
            indicesB[b->ndim - 1] = k;
            int idxR = t_get_flat_index(r, stridesR, indicesR);
            int idxB = t_get_flat_index(b, stridesB, indicesB);
            r->v[idxR] += a->v[flatIdx] * b->v[idxB];
        }
    }

    if (originalA->ndim == 1)
        t_free(a);
    if (originalB->ndim == 1)
        t_free(b);

    return r;
}

tensor *__t_transpose_2dim(tensor *t)
{
    // all variables ending with `T` are the transposed variant.
    int shapeT[t->ndim], idc[t->ndim], idcT[t->ndim], strides[t->ndim], stridesT[t->ndim];
    int unchangedDimensions = t->ndim - 2;
    // first, copy all
    __t_copy_shape(t, unchangedDimensions, shapeT);

    shapeT[t->ndim - 1] = t->shape[t->ndim - 2];
    shapeT[t->ndim - 2] = t->shape[t->ndim - 1];

    tensor *tT = t_alloc(t->ndim, shapeT);

    t_calc_strides(t, strides);
    t_calc_strides(tT, stridesT);

    for (int flatIdx = 0; flatIdx < t->_v_size; flatIdx++)
    {
        t_get_indices(t, flatIdx, strides, idc);
        __copy_array(idc, idcT, unchangedDimensions);

        // do the permutation
        idcT[t->ndim - 1] = idc[t->ndim - 2];
        idcT[t->ndim - 2] = idc[t->ndim - 1];

        // store the value in the transposed tensor
        int flatIdxT = t_get_flat_index(tT, stridesT, idcT);
        tT->v[flatIdxT] = t->v[flatIdx];
    }
    return tT;
}

tensor *__t_transpose_1dim(tensor *t)
{
    tensor *with_added_dim = t_add_dim(t);
    tensor *tT = __t_transpose_2dim(with_added_dim);
    t_free(with_added_dim);
    return tT;
}

tensor *t_transpose(tensor *t, int transposeLastNDimensions)
{
    /**
     * Transposes a tensors last n dimensions.
     */
    switch (transposeLastNDimensions)
    {
    case 2:
        return __t_transpose_2dim(t);
    case 1:
        return __t_transpose_1dim(t);
    default:
        error("Can only transpose the up to the last two dimensions at the moment, tried %i.", transposeLastNDimensions);
        return NULL; // unreachable
    }
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
        return t_alloc_single_from(t_collapse_sum_all(t));
    }

    if (collapseDimIdx < 0)
        collapseDimIdx = t->ndim - 1;
    else if (collapseDimIdx >= t->ndim)
        error("Trying to collapse across non-existing dimension.");

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
        int dimsSize[t->ndim];
        for (int i = t->ndim - 1; i >= 0; i--)
        {
            if (i >= t->ndim - 1)
                dimsSize[i] = t->shape[i];
            else
                dimsSize[i] = t->shape[i] * dimsSize[i + 1];
        }

        for (int d = 0; d < t->ndim; d++)
        {
            int d_idx = -1;
            // Calculate dimension indicies of the old matrix
            if (d < t->ndim - 1)
            {
                d_idx = rem / dimsSize[d + 1];
                rem -= d_idx * dimsSize[d + 1];
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

param_t t_collapse_sum_all(tensor *t)
{
    param_t sum = 0;
    for (int i = 0; i < t->_v_size; i++)
    {
        sum += t->v[i];
    }
    return sum;
}

param_t t_collapse_mean_all(tensor *t)
{
    param_t sum = 0;
    for (int i = 0; i < t->_v_size; i++)
    {
        sum += t->v[i];
    }
    return sum / t->_v_size;
}

/**
 * Utils
 */

void t_print_shape(tensor *t)
{
    printf("[");
    for (int i = 0; i < t->ndim; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%i", t->shape[i]);
    }
    printf("]");
}

void t_print(tensor *t)
{
    if (t->ndim == 1 && t->shape[0] == 1)
    {
        // single element tensor
        printf("%.4f", t->v[0]);
        return;
    }

    // this will store the multidimensional coords derived from singledimensional coords
    // based on the shape/offsets.
    int *currentDimsIdx = mm_calloc(t->ndim, sizeof(int));

    // Opening brackets for all dimensions
    for (int d = 0; d < t->ndim; d++)
        printf("[");

    int closingStack = 0; // number of brackets we are closing right now (and need to reopen as well)
    for (int i = 0; i < t->_v_size; i++)
    {
        for (int d = t->ndim - 1; d >= 0; d--)
        {
            int oldDimIdx = currentDimsIdx[d];
            if (i > 0)
                currentDimsIdx[d] = (currentDimsIdx[d] + 1) % t->shape[d];
            if (oldDimIdx < currentDimsIdx[d])
            {
                // we are still moving along the same dimension,
                // so no need to check the other dimensions.
                break;
            }
            else
            {
                // New dimension. We will continue the loop, but now we will close & open
                // the brackets of the dimension
                if (i != 0)
                    closingStack++;
            }
        }

        // Close & reopen brackets, if needed
        if (closingStack > 0)
        {
            for (int c = 0; c < closingStack; c++)
                printf("]");
            printf("\n");
            for (; closingStack > 0; closingStack--) // resetting closing stack
                printf("[");
        }
        else
        {
            if (i > 0)
                printf(", ");
        }

        // Now, finally we can print our numbers
        printf("%.4f", t->v[i]);
    }
    // Closing brackets for all dimensions.
    // No need for a closing stack, as we are always all dimensions deep here
    for (int d = 0; d < t->ndim; d++)
        printf("]");
    printf("\n");
    mm_free(currentDimsIdx);
}