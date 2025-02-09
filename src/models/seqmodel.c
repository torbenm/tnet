#include <string.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"

struct seqmodel *seqmodel_init()
{
    struct seqmodel *seq = mm_alloc(sizeof(struct seqmodel));
    seq->_layerBufferSize = 0;
    seqmodel_resize(seq, SEQMODEL_STD_SIZE);
    return seq;
}

void seqmodel_dodge(struct seqmodel *seq)
{
    mm_dodge(seq);
    mm_dodge(seq->layers);
    for (int l = 0; l < seq->numLayers; l++)
    {
        seq->layers[l]->dodge(seq->layers[l]);
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

tensor *seqmodel_predict(struct seqmodel *seq, tensor *input)
{
    tensor *next_inputs = input;
    for (int i = 0; i < seq->numLayers; i++)
    {
        tensor *output = seq->layers[i]->forward(seq->layers[i]->layerProps, next_inputs, NULL);
        if (i > 1) // making sure we are not freeing the original inputs
            t_free(next_inputs);
        next_inputs = output;
    }
    return next_inputs;
}