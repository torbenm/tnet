#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "tnet.h"
#include "mat.h"

void layer_init(struct layer *l, int numNodes, int numInputs, activationfunc *activationFn)
{
    l->numNodes = numNodes;
    l->numInputs = numInputs;
    l->weights = mat_alloc(numNodes, numInputs);
    l->activationFn = activationFn;
}

param_t *layer_forward(struct layer *l, vec inputs)
{
    vec output = mat_dot_product(l->weights, inputs, l->numNodes, l->numInputs);
    vec_apply_ip(output, l->activationFn, l->numNodes);
    return output;
}

void layer_free(struct layer *l)
{
    mat_free(l->weights, l->numNodes);
    free(l);
}
