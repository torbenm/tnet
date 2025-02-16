#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

typedef struct csv_reader
{
    FILE *file;
} csv_reader;

typedef struct data_header
{
    int numInputColumns;
    int numTruthColumns;
    char **inputColumns;
    char **truthColumns;
} data_header;

csv_reader *csv_open(const char *filename);
void csv_close(csv_reader *c);
int csv_seek_next_line(csv_reader *c);
char *csv_next_field(csv_reader *c);
int csv_next_field_int(csv_reader *c);
param_t str_to_param(const char *field);

struct seqmodel *seqmodel_from_file(const char *filename);
struct seqmodel_layer *seqmodel_layer_from_csv(const char *layer_type, csv_reader *c);
activationfunc *activationfunc_from_str(const char *name);

tensor *tensor_from_csv_1dim(csv_reader *c);
tensor **tensor_array_from_file(const char *filename, int *outNumRows);
int parse_csv_into_inputs_and_truth(const char *filename, tensor ***outInputs, tensor ***outtruth, int truthInputSplitIdx, data_header *header);
void print_data_header(data_header *header);