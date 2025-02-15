#include <stdio.h>

#include "core.h"
#include "funcs.h"
#include "models.h"

typedef struct csv_reader
{
    FILE *file;
} csv_reader;

csv_reader *csv_open(const char *filename);
void csv_close(csv_reader *c);
int csv_seek_next_line(csv_reader *c);
char *csv_next_field(csv_reader *c);
int csv_next_field_int(csv_reader *c);
param_t csv_next_field_param(csv_reader *c);

// test
void test_csv();

struct seqmodel *seqmodel_from_file(const char *filename);
struct seqmodel_layer *seqmodel_layer_from_csv(const char *layer_type, csv_reader *c);
activationfunc *activationfunc_from_str(const char *name);