#include <stdio.h>
#include <stdlib.h>

#include "core.h"
#include "models.h"

struct backwardstate *backwardstate_alloc()
{
    return mm_alloc(sizeof(struct backwardstate));
}

void backwardstate_mark(struct backwardstate *b)
{
    if (b == NULL)
        return;
    mm_mark(b);
    t_mark(b->biasGradients);
    t_mark(b->weightGradients);
    t_mark(b->smallDelta);
}

void backwardstate_free(struct backwardstate *i)
{
    if (i == NULL)
        return;
    t_free(i->biasGradients);
    t_free(i->weightGradients);
    t_free(i->smallDelta);
    mm_free(i);
}

void backwardstate_lock(struct backwardstate *b)
{
    if (b == NULL)
        return;
    t_lock(b->biasGradients);
    t_lock(b->weightGradients);
    t_lock(b->smallDelta);
}