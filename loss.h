#include "tnet.h"

typedef param_t lossfunc(int numExamples, param_t predictions[numExamples], param_t truths[numExamples]);

lossfunc loss_mse;
lossfunc loss_abssum;