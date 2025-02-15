#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "core.h"
#include "models.h"

#define SEQMODEL_STD_SIZE 4

struct seqmodel *seqmodel_init()
{
    struct seqmodel *seq = mm_alloc(sizeof(struct seqmodel));
    seq->_layerBufferSize = 0;
    seqmodel_resize(seq, SEQMODEL_STD_SIZE);
    return seq;
}

void seqmodel_mark(struct seqmodel *seq)
{
    mm_mark(seq);
    mm_mark(seq->layers);
    for (int l = 0; l < seq->numLayers; l++)
    {
        seqmodel_layer_mark(seq->layers[l]);
    }
}

void seqmodel_resize(struct seqmodel *seq, int newSize)
{
    struct seqmodel_layer **new_layers = mm_alloc((newSize) * sizeof(struct seqmodel_layer *));
    if (seq->_layerBufferSize > 0)
    {
        memcpy(new_layers, seq->layers, seq->_layerBufferSize);
        mm_free(seq->layers);
    }
    seq->_layerBufferSize = newSize;
    seq->layers = new_layers;
}

void seqmodel_push(struct seqmodel *seq, struct seqmodel_layer *layer)
{
    if (seq->numLayers == seq->_layerBufferSize)
        seqmodel_resize(seq, seq->_layerBufferSize * 2);

    seq->layers[seq->numLayers++] = layer;
}

void seqmodel_print(struct seqmodel *s)
{
    for (int l = 0; l < s->numLayers; l++)
    {
        seqmodel_layer_print(s->layers[l]);
    }
}

tensor *seqmodel_predict(struct seqmodel *seq, tensor *input)
{
    tensor *next_inputs = input;
    for (int i = 0; i < seq->numLayers; i++)
    {
        tensor *output = seq->layers[i]->forward(seq->layers[i], next_inputs, NULL);
        if (i > 1) // making sure we are not freeing the original inputs
            t_free(next_inputs);
        next_inputs = output;
    }
    return next_inputs;
}

param_t seqmodel_calculate_loss(struct seqmodel *seq, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], lossfunc *lossFn)
{
    tensor *predictions[batchSize];
    for (int t = 0; t < batchSize; t++)
    {
        predictions[t] = seqmodel_predict(seq, inputs[t]);
    }
    return lossFn(batchSize, predictions, truths);
}