#pragma once

#include "mat.h"

struct intermediate
{
    vec activations;
    vec preActivations;
    int nOutputs;
};

struct intermediate *intermediate_alloc(int size);
void intermediate_free(struct intermediate *);