#include "core.h"
#include "models.h"
#include "funcs.h"
#include "train.h"
#include <stdio.h>

#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 100000
#define LEARNING_RATE 0.1
#define MONUMENTUM 0.9

void seq_example_XOR()
{
    const int numLayers = 1;
    int numParamsPerLayer[1] = {2};

    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

    tensor *t_values[4] = {
        t_from_1dim_array(2, values[0]),
        t_from_1dim_array(2, values[1]),
        t_from_1dim_array(2, values[2]),
        t_from_1dim_array(2, values[3]),
    };

    tensor *t_truths[4] = {
        t_from_1dim_array(2, truths[0]),
        t_from_1dim_array(2, truths[1]),
        t_from_1dim_array(2, truths[2]),
        t_from_1dim_array(2, truths[3]),
    };

    struct seqmodel *s = seqmodel_init();
    seqmodel_push(s, input_layer_init(2));
    seqmodel_push(s, dense_layer_init(2, 2, av_tanh));
    seqmodel_push(s, dense_layer_init(2, 2, av_softmax));

    for (int i = 0; i < 4; i++)
    {
        t_print(seqmodel_predict(s, t_from_1dim_array(2, values[i])));
    }

    struct optimizer *o = opt_sgd_init(LEARNING_RATE, MONUMENTUM, loss_mse);
    train(s, 4, t_values, t_truths, MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    for (int i = 0; i < 4; i++)
    {
        t_print(seqmodel_predict(s, t_from_1dim_array(2, values[i])));
    }
}