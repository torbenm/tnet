#include "tnet.h"
#include "layer.h"
#include "activation.h"
#include "loss.h"

struct mlp
{
    int numLayers;
    int numInputs;
    int numOutputs;
    struct layer **layers;
    lossfunc *lossFn;
};

struct mlp *mlp_init(int numInputs, int numOutputs, int numHiddenLayers, int numParams[numHiddenLayers]);
param_t *mlp_forward(struct mlp *p, param_t values[p->numInputs]);
