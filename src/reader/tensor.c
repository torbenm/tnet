#include <stdio.h>
#include <stdlib.h>

#include "reader.h"
#include "core.h"

#define VALUE_BUFFER_SIZE 1024
#define TENSOR_BUFFER_SIZE 1024

/**
 * Reads a 1-dim tensor (vector) from a csv file.
 * Expected format: <value_1>;<value_2;...;<value_n>
 *
 * At most VALUE_BUFFER_SIZE values.
 */
tensor *tensor_from_csv_1dim(csv_reader *c)
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

tensor **tensor_array_from_file(const char *filename, int *outNumRows)
{
    tensor *buffer[TENSOR_BUFFER_SIZE];
    int buf_ptr = 0;
    *outNumRows = 0;

    csv_reader *c = csv_open(filename);
    // will run seek next line at first, skipping header line
    char *layer_type;
    while (csv_seek_next_line(c) != EOF)
    {
        buffer[buf_ptr++] = tensor_from_csv_1dim(c);
    }
    tensor **out = mm_alloc(buf_ptr * sizeof(tensor *));

    for (int i = 0; i < buf_ptr; i++)
        out[i] = buffer[i];

    csv_close(c);
    *outNumRows = buf_ptr;
    return out;
}