#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "reader.h"
#include "core.h"

#define VALUE_BUFFER_SIZE 1024
#define TENSOR_BUFFER_SIZE 1024

/**
 * Reads a 1-dim tensor (vector) from a csv file with a flexible size.
 * Expected format: <value_1>;<value_2;...;<value_n>
 *
 * At most VALUE_BUFFER_SIZE values.
 */
tensor *tensor_from_csv_1dim_var(csv_reader *c)
{
    param_t buffer[VALUE_BUFFER_SIZE];
    int buf_ptr = 0;
    char *next = csv_next_field(c);
    while (next != NULL && buf_ptr < VALUE_BUFFER_SIZE)
    {
        buffer[buf_ptr++] = str_to_param(next);
        next = csv_next_field(c);
    }
    return t_from_1dim_array(buf_ptr, buffer);
}

/**
 * Reads a 1-dim tensor (vector) from a csv file with a fixed size.
 * Expected format: <value_1>;<value_2;...;<value_n>
 *
 * At most VALUE_BUFFER_SIZE values.
 */
tensor *tensor_from_csv_1dim_fixed(csv_reader *c, int size)
{
    param_t buffer[size];
    for (int i = 0; i < size; i++)
    {
        buffer[i] = str_to_param(csv_next_field(c));
    }
    return t_from_1dim_array(size, buffer);
}

tensor **tensor_array_from_file(const char *filename, int *outNumRows)
{
    tensor *buffer[TENSOR_BUFFER_SIZE];
    int buf_ptr = 0;
    *outNumRows = 0;

    csv_reader *c = csv_open(filename);
    while (csv_seek_next_line(c) != EOF)
    {
        buffer[buf_ptr++] = tensor_from_csv_1dim_var(c);
    }
    tensor **out = mm_alloc(buf_ptr * sizeof(tensor *));

    for (int i = 0; i < buf_ptr; i++)
        out[i] = buffer[i];

    csv_close(c);
    *outNumRows = buf_ptr;
    return out;
}

int parse_csv_row_into_two_string_sets(csv_reader *c, int splitIndex, char ***outA, char ***outB)
{
    char *buffer[VALUE_BUFFER_SIZE];
    int buf_ptr = 0;
    char *next = csv_next_field(c);
    while (next != NULL && buf_ptr < VALUE_BUFFER_SIZE)
    {
        buffer[buf_ptr++] = next;
        next = csv_next_field(c);
    }
    if (buf_ptr < splitIndex)
        error("Found too little values - expected at least %i, got %i.", splitIndex + 1, buf_ptr);

    *outA = mm_alloc(sizeof(char *) * splitIndex);
    for (int i = 0; i < splitIndex; i++)
    {
        (*outA)[i] = buffer[i];
    }

    *outB = mm_alloc(sizeof(char *) * (buf_ptr - splitIndex));
    for (int i = splitIndex; i < buf_ptr; i++)
    {
        (*outB)[i - splitIndex] = buffer[i];
    }
    return buf_ptr;
}

int parse_csv_into_inputs_and_truth(const char *filename, tensor ***outInputs, tensor ***outTruth, int truthInputSplitIdx, data_header *outHeader)
{
    csv_reader *c = csv_open(filename);

    // Read Header
    int totalNumColumns = parse_csv_row_into_two_string_sets(c, truthInputSplitIdx, &outHeader->inputColumns, &outHeader->truthColumns);
    outHeader->numInputColumns = truthInputSplitIdx;
    outHeader->numTruthColumns = totalNumColumns - truthInputSplitIdx;

    // Read Values
    tensor *inputsBuffer[TENSOR_BUFFER_SIZE];
    tensor *truthBuffer[TENSOR_BUFFER_SIZE];
    int bufPtr = 0;
    for (bufPtr = 0; csv_seek_next_line(c) != EOF && bufPtr < TENSOR_BUFFER_SIZE; bufPtr++)
    {
        inputsBuffer[bufPtr] = tensor_from_csv_1dim_fixed(c, outHeader->numInputColumns);
        truthBuffer[bufPtr] = tensor_from_csv_1dim_fixed(c, outHeader->numTruthColumns);
    }

    *outInputs = mm_alloc(bufPtr * sizeof(tensor *));
    *outTruth = mm_alloc(bufPtr * sizeof(tensor *));

    for (int i = 0; i < bufPtr; i++)
    {
        (*outInputs)[i] = inputsBuffer[i];
        (*outTruth)[i] = truthBuffer[i];
    }

    csv_close(c);
    return bufPtr;
}