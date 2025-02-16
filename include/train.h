#include "core.h"
#include "models.h"

struct trainingpass
{
    tensor **stored_tensors;
    param_t loss;
    int numStoredTensors;
};

struct trainingpass *trainingpass_init(param_t loss, tensor **stored_tensors, int numLayers);
void trainingpass_free(struct trainingpass *tp);
void trainingpass_lock(struct trainingpass *tp);
void trainingpass_mark(struct trainingpass *tp);

typedef struct trainingpass *opt_func(struct seqmodel *seq, param_t *params, int numExamples, tensor *inputs[numExamples], tensor *truths[numExamples], loss *loss, struct trainingpass *previouspass, int trainingPassNum);

typedef struct optimizer
{
    int numParams;
    char *name;
    param_t *params;
    opt_func *run_opt;
    loss *loss;
} optimizer;

void opt_mark(struct optimizer *o);
struct forwardstate *opt_forwardpropagate(struct seqmodel *seq, tensor *inputs, tensor **outPredictions);
struct backwardstate **opt_backwardpropagate(struct seqmodel *seq, tensor *prediction, tensor *truth, struct forwardstate *forwardstates, loss *loss);
void opt_fowardbackwardpass(struct seqmodel *s, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], tensor ***outWeightGradients, tensor ***outBiasGradients, loss *loss);
param_t opt_calculateloss(struct seqmodel *seq, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], loss *loss);

// SGD
struct optimizer *opt_sgd_init(param_t learningRate, param_t monumentum, loss *loss);
opt_func opt_sgd;

// Adam
struct optimizer *opt_adam_init(param_t alpha, param_t beta1, param_t beta2, loss *loss);
opt_func opt_adam;

// Training function
void train(struct seqmodel *seq, int numExamples, tensor *inputs[numExamples], tensor *truths[numExamples], int maxIter, optimizer *opt, param_t diffThreshold, param_t lossThreshold);
void check_gradients(struct seqmodel *s, int batchSize, tensor *inputs[batchSize], tensor *truths[batchSize], loss *loss);