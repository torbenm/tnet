#include "core.h"
#include "models.h"

typedef param_t opt_func(struct seqmodel *seq, param_t *params, int numExamples, vec inputs[numExamples], vec truths[numExamples], lossfunc *lossFn);
struct optimizer
{
    int numParams;
    param_t *params;
    opt_func *run_opt;
    lossfunc *lossFn;
};

void train(struct seqmodel *seq, int numExamples, vec inputs[numExamples], vec truths[numExamples], int maxIter, struct optimizer *opt, param_t diffThreshold, param_t lossThreshold);

struct optimizer *opt_sgd_init(param_t learningRate, lossfunc *lossFn);
opt_func opt_sgd;