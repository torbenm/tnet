#include "tnet.h"
#include "mat.h"

typedef param_t lossfunc(int numExamples, int vecSize, vec predictions[numExamples], vec truths[numExamples]);

lossfunc loss_mse;
lossfunc loss_abssum;