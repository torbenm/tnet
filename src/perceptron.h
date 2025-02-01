#include "tnet.h"
#include "activation.h"
#include "mat.h"

struct perceptron
{
    param_t bias;
    vec weights;
    int numWeights;
    activationfunc *activationFn;
};

vec perceptron_forward(struct perceptron *p, vec vals);
int perceptron_train(struct perceptron *p, int maxIter, int numTVals, mat tvals, vec truth, param_t learningRate);
struct perceptron *perceptron_init(int numWeights);
void perceptron_free(struct perceptron *p);
void perceptron_forward_print(struct perceptron *p, int numVals, mat values, vec truth);