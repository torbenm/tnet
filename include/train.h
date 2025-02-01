#include "core.h"
#include "models.h"

void opt_gradient_descent(int maxIter, param_t learningRate, param_t costThreshold, param_t diffThreshold, mat trainingValues, int numTrainingValues, mat truths, struct mlp *p);