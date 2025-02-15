#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models.h"
#include "reader.h"

struct seqmodel *seqmodel_from_file(const char *filename)
{
    struct seqmodel *s = seqmodel_init();
    csv_reader *c = csv_open(filename);

    // will run seek next line at first, skipping header line
    char *layer_type;
    while (csv_seek_next_line(c) != EOF)
    {
        layer_type = csv_next_field(c);
        if (layer_type != NULL)
        {
            seqmodel_push(s, seqmodel_layer_from_csv(layer_type, c));
        }
    }

    csv_close(c);
    return s;
}

struct seqmodel_layer *seqmodel_layer_from_csv(const char *layer_type, csv_reader *c)
{
    if (strcmp(layer_type, "INPUT") == 0)
    {
        return input_layer_init(csv_next_field_int(c));
    }
    if (strcmp(layer_type, "DENSE") == 0)
    {
        return dense_layer_init(
            csv_next_field_int(c),
            csv_next_field_int(c),
            activationfunc_from_str(csv_next_field(c)));
    }
    if (strcmp(layer_type, "SOFTMAX") == 0)
    {
        return softmax_layer_init();
    }
    return NULL;
}

activationfunc *activationfunc_from_str(const char *name)
{
    if (strcmp(name, "TANH"))
        return av_tanh;
    if (strcmp(name, "IDENTITY"))
        return av_identity;
    if (strcmp(name, "SIGMOID") || strcmp(name, "LOGISTIC"))
        return av_sigmoid; // same as logistic
    if (strcmp(name, "RELU"))
        return av_relu;
    return NULL;
}
