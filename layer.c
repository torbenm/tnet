#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "tnet.h"
#include "intermediate.h"
#include "mat.h"

struct layer *layer_init(int numNodes, int numInputs, activationfunc *activationFn)
{
    struct layer *l = malloc(sizeof(struct layer));
    l->numNodes = numNodes;
    l->numInputs = numInputs;
    l->weights = mat_alloc_rand(numNodes, numInputs);
    l->bias = vec_alloc_rand(numNodes);
    l->activationFn = activationFn;
    return l;
}

vec layer_forward(struct layer *l, vec inputs, struct intermediate *i)
{
    // inputs: inputs x 1
    // l->weights: inputs x nodes
    // preActivations, activations: nodes x 1
    i->nOutputs = l->numNodes;
    vec dot_prod = mat_dot_product(l->weights, inputs, l->numNodes, l->numInputs);
    i->preActivations = vec_elem_add(dot_prod, l->bias, l->numNodes);
    i->activations = l->activationFn(i->preActivations, l->numNodes, ACTIVATION_FORWARD);
    vec_free(dot_prod);
    return i->activations;
}

/**
 * previousSmallDelta: 1 x nodes
 */
vec layer_backward(struct layer *l, vec previousSmallDelta, param_t learningRate, struct intermediate *curr, struct intermediate *prev, int isOutputLayer)
{
    setbuf(stdout, NULL);

    // printf("\nNumNodes = %i; NumInputs = %i;\n", l->numNodes, l->numInputs);
    // printf("PrevSmallDelta = ");
    // vec_print(previousSmallDelta, l->numNodes);
    // printf("\n");
    // printf("old_weights=");
    // mat_print(l->weights, l->numNodes, l->numInputs);
    // printf("\n");
    // printf("preActivations=");
    // vec_print(curr->preActivations, l->numNodes);
    // printf("\n");
    // printf("prevActivations=");
    // vec_print(prev->activations, l->numInputs);
    // printf("\n--------------\n");

    vec smallDelta = previousSmallDelta;
    if (!isOutputLayer)
    {
        // Middle layer - previousSmallDelta needs to be multiplied with activation derivation
        vec actDeriv = l->activationFn(curr->preActivations, l->numNodes, ACTIVATION_DERIVATIVE);
        smallDelta = vec_elem_mul(smallDelta, actDeriv, l->numNodes);
        // printf("actDeriv=");
        // vec_print(actDeriv, l->numNodes);
        // printf("\n");
        // printf("smallDelta=");
        // vec_print(smallDelta, l->numNodes);
        // printf("\n");
        // vec_free(actDeriv);
    }

    // calculate nextSmallDelta
    mat weights_t = mat_transpose(l->weights, l->numNodes, l->numInputs);
    vec nextSmallDelta = mat_dot_product(weights_t, smallDelta, l->numInputs, l->numNodes);
    // printf("nextSmallDelta=");
    // vec_print(nextSmallDelta, l->numInputs);
    // printf("\n");

    // why positive 1.0? LEARNING_RATE missing here
    mat deltaW = vec_transposed_vec_mul(smallDelta, prev->activations, 1.0 * learningRate, l->numNodes, l->numInputs);
    mat newWeights = mat_elem_add(l->weights, deltaW, l->numNodes, l->numInputs);

    // printf("deltaW=");
    // mat_print(deltaW, l->numNodes, l->numInputs);
    // printf("\n");
    // printf("new_weights=");
    // mat_print(newWeights, l->numNodes, l->numInputs);
    // printf("\n");
    mat_free(l->weights, l->numNodes);
    l->weights = newWeights;

    // vec deltaB = vec_mul_const(smallDelta, -1, l->numNodes);
    vec newB = vec_elem_add(l->bias, smallDelta, l->numNodes);
    vec_free(l->bias);
    l->bias = newB;

    mat_free(weights_t, l->numInputs);
    mat_free(deltaW, l->numNodes);

    return nextSmallDelta;
}

void layer_free(struct layer *l)
{
    mat_free(l->weights, l->numNodes);
    free(l);
}
