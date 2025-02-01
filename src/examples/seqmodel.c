#include "core.h"
#include "models.h"
#include "funcs.h"
#include "train.h"

#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 100000
#define LEARNING_RATE 0.1

void seq_example_XOR()
{
    const int numLayers = 1;
    int numParamsPerLayer[1] = {2};

    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

    struct seqmodel *s = seqmodel_init();
    seqmodel_push(s, input_layer_init(2));
    seqmodel_push(s, dense_layer_init(2, 2, av_tanh));
    seqmodel_push(s, dense_layer_init(2, 2, av_softmax));

    for (int i = 0; i < 4; i++)
    {
        vec_print(seqmodel_predict(s, vec_from_array(2, values[i])), 2);
    }

    struct optimizer *o = opt_sgd_init(LEARNING_RATE, loss_mse);
    train(s, 4, mat_from_array(4, 2, values), mat_from_array(4, 2, truths), MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    for (int i = 0; i < 4; i++)
    {
        vec_print(seqmodel_predict(s, vec_from_array(2, values[i])), 2);
    }
}