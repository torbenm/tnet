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
    int bufPtr = 0;
    int lastReadStatus = READ_STATUS_MORE;
    char *next = csv_next_field(c, &lastReadStatus);
    while (lastReadStatus == READ_STATUS_MORE && bufPtr < VALUE_BUFFER_SIZE)
    {
        buffer[bufPtr++] = str_to_param(next);
        next = csv_next_field(c, &lastReadStatus);
    }
    return t_from_1dim_array(bufPtr, buffer);
}

/**
 * Reads a 1-dim tensor (vector) from a csv file with a fixed size.
 * Expected format: <value_1>;<value_2;...;<value_n>
 *
 * At most VALUE_BUFFER_SIZE values.
 */
tensor *tensor_from_csv_1dim_fixed(csv_reader *c, int size, int *outReadStatus)
{
    param_t buffer[size];
    for (int i = 0; i < size && *outReadStatus == READ_STATUS_MORE; i++)
    {
        buffer[i] = str_to_param(csv_next_field(c, outReadStatus));
    }
    return t_from_1dim_array(size, buffer);
}

tensor **tensor_array_from_file(const char *filename, int *outNumRows)
{
    tensor *buffer[TENSOR_BUFFER_SIZE];
    int bufPtr = 0;
    *outNumRows = 0;

    csv_reader *c = csv_open(filename);
    while (csv_seek_next_line(c, -1) != EOF)
    {
        buffer[bufPtr++] = tensor_from_csv_1dim_var(c);
    }
    tensor **out = mm_alloc(bufPtr * sizeof(tensor *));

    for (int i = 0; i < bufPtr; i++)
        out[i] = buffer[i];

    csv_close(c);
    *outNumRows = bufPtr;
    return out;
}

int parse_csv_row_into_two_string_sets(csv_reader *c, int splitIndex, char ***outA, char ***outB)
{
    char *buffer[VALUE_BUFFER_SIZE];
    int bufPtr = 0;
    int lastReadStatus = READ_STATUS_MORE;
    char *next;
    do
    {
        buffer[bufPtr++] = csv_next_field(c, &lastReadStatus);
    } while (lastReadStatus == READ_STATUS_MORE && bufPtr < VALUE_BUFFER_SIZE);
    if (bufPtr <= splitIndex)
        error("Found too little values - expected at least %i, got %i.", splitIndex + 1, bufPtr);

    *outA = mm_alloc(sizeof(char *) * splitIndex);
    for (int i = 0; i < splitIndex; i++)
    {
        (*outA)[i] = buffer[i];
    }
    int outBsize = bufPtr - splitIndex;
    *outB = mm_calloc(outBsize, sizeof(char *));
    for (int i = splitIndex; i < bufPtr; i++)
    {
        (*outB)[i - splitIndex] = buffer[i];
    }
    return bufPtr;
}

int parse_csv_into_inputs_and_truth(const char *filename, tensor ***outInputs, tensor ***outTruth, int truthInputSplitIdx, data_header *header)
{
    csv_reader *c = csv_open(filename);
    // Read Header
    int totalNumColumns = parse_csv_row_into_two_string_sets(c, truthInputSplitIdx, &(header->inputColumns), &(header->truthColumns));
    header->numInputColumns = truthInputSplitIdx;
    header->numTruthColumns = totalNumColumns - truthInputSplitIdx;
    print_data_header(header);

    // Read Values
    tensor *inputsBuffer[TENSOR_BUFFER_SIZE];
    tensor *truthBuffer[TENSOR_BUFFER_SIZE];
    int bufPtr = 0;
    int lastReadStatus;
    do
    {
        lastReadStatus = READ_STATUS_MORE;

        inputsBuffer[bufPtr] = tensor_from_csv_1dim_fixed(c, header->numInputColumns, &lastReadStatus);
        truthBuffer[bufPtr++] = tensor_from_csv_1dim_fixed(c, header->numTruthColumns, &lastReadStatus);

        lastReadStatus = csv_seek_next_line(c, lastReadStatus);
    } while (lastReadStatus != READ_STATUS_EOF && bufPtr < TENSOR_BUFFER_SIZE);

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

void print_data_header(data_header *header)
{
    printf("Inputs (%i): ", header->numInputColumns);
    for (int i = 0; i < header->numInputColumns; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%s", header->inputColumns[i]);
    }
    printf("\nTruths (%i): ", header->numTruthColumns);
    for (int i = 0; i < header->numTruthColumns; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%s", header->truthColumns[i]);
    }
    printf("\n");
}