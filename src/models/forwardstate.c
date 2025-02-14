#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"

struct forwardstate *forwardstate_alloc()
{
    struct forwardstate *i = mm_alloc(sizeof(struct forwardstate));
    return i;
}

void forwardstate_free(struct forwardstate *i)
{
    t_free(i->activations);
    t_free(i->preActivations);
    mm_free(i);
}

void forwardstate_mark(struct forwardstate *f)
{
    mm_mark(f);
    t_mark(f->activations);
    t_mark(f->preActivations);
}

void forwardstate_lock(struct forwardstate *f)
{
    t_lock(f->activations);
    t_lock(f->preActivations);
}