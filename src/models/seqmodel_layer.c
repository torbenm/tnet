#include <stdio.h>

#include "models.h"

struct backwardstate *seqmodel_layer_backward_noop(void *p, tensor *previousSmallDelta, struct forwardstate *curr, struct forwardstate *prev)
{
    return NULL; // should _not_ be used!
}

void seqmodel_layer_update_noop(void *p, tensor *updateWeights, tensor *updateBias)
{
    // NOOP
}

struct seqmodel_layer *seqmodel_layer_init(void *props, const char *name, seqmodel_layer_forward *forward, seqmodel_layer_backward *backward, seqmodel_layer_update_weights *update, void (*markProps)(void *))
{
    // move somewhere else...
    struct seqmodel_layer *l = mm_alloc(sizeof(struct seqmodel_layer));
    l->layerProps = props;
    l->name = name;

    if (forward == NULL)
        error("Must provide forward function for seqmodel_layer.");

    // defaults
    l->update = seqmodel_layer_update_noop;
    l->backward = seqmodel_layer_backward_noop;
    l->markProps = NULL;
    l->forward = forward; // must be implemented!

    if (backward != NULL)
        l->backward = backward;
    if (update != NULL)
        l->update = update;
    if (markProps != NULL)
        l->markProps = markProps;
    return l;
}

void seqmodel_layer_print(struct seqmodel_layer *self)
{
    printf("<layer name='%s' />\n", self->name);
}

void seqmodel_layer_mark(struct seqmodel_layer *self)
{
    if (self->layerProps != NULL)
    {
        mm_mark(self->layerProps);

        if (self->markProps != NULL)
            self->markProps(self->layerProps);
    }
    mm_mark(self);
}
