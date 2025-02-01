#include "core.h"
#include "models.h"

struct trainingpass
{
    struct backwardstate *backwardstates;
    param_t loss;
    int numLayers;
};

struct trainingpass *trainingpass_init(param_t loss, struct backwardstate *backwardstates, int numLayers);
void trainingpass_free(struct trainingpass *tp);

typedef struct trainingpass *opt_func(struct seqmodel *seq, param_t *params, int numExamples, vec inputs[numExamples], vec truths[numExamples], lossfunc *lossFn, struct trainingpass *previouspass);
struct optimizer
{
    int numParams;
    param_t *params;
    opt_func *run_opt;
    lossfunc *lossFn;
};

void train(struct seqmodel *seq, int numExamples, vec inputs[numExamples], vec truths[numExamples], int maxIter, struct optimizer *opt, param_t diffThreshold, param_t lossThreshold);

struct optimizer *opt_sgd_init(param_t learningRate, param_t monumentum, lossfunc *lossFn);
opt_func opt_sgd;
