#include <stdio.h>

#include "models.h"

struct backwardstate *seqmodel_layer_backward_noop(struct seqmodel_layer *self, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    return NULL; // should _not_ be used!
}

void seqmodel_layer_update_noop(struct seqmodel_layer *self, tensor *updateWeights, tensor *updateBias)
{
    // weights = weights - (weightGradients * updateFactor)
    t_elem_sub(self->weights, updateWeights);

    // bias = bias - (biasGradients * updateFactor)
    t_elem_sub(self->bias, updateBias);
}

struct seqmodel_layer *seqmodel_layer_init(const char *name, tensor *weights, tensor *bias, activationfunc *activationFn, seqmodel_layer_forward *forward, seqmodel_layer_backward *backward)
{
    // move somewhere else...
    struct seqmodel_layer *l = mm_alloc(sizeof(struct seqmodel_layer));
    l->name = name;

    if (forward == NULL)
        error("Must provide forward function for seqmodel_layer.");

    // defaults
    l->update = seqmodel_layer_update_noop;
    l->backward = seqmodel_layer_backward_noop;
    l->forward = forward; // must be implemented!
    l->activationFn = activationFn;

    if (weights != NULL)
        l->weights = weights;
    else
        l->weights = t_null();

    if (bias != NULL)
        l->bias = bias;
    else
        l->bias = t_null();

    if (backward != NULL)
        l->backward = backward;
    return l;
}

void seqmodel_layer_print(struct seqmodel_layer *self)
{
    printf("<layer name='%s'", self->name);
    if (self->weights->ndim > 0)
    {
        printf(" weights=");
        t_print_shape(self->weights);
    }

    if (self->bias->ndim > 0)
    {
        printf(" bias=");
        t_print_shape(self->bias);
    }
    printf(" />\n");
}

void seqmodel_layer_mark(struct seqmodel_layer *self)
{
    t_mark(self->bias);
    t_mark(self->weights);
    mm_mark(self);
}
