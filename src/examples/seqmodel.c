#include "core.h"
#include "models.h"
#include "funcs.h"
#include "train.h"
#include <stdio.h>

#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 10000000
#define LEARNING_RATE 0.1
#define MONUMENTUM 0.9

void seq_example_EXEC(param_t values[4][2], param_t truths[4][2])
{

    tensor *t_values[4] = {
        t_lock(t_from_1dim_array(2, values[0])),
        t_lock(t_from_1dim_array(2, values[1])),
        t_lock(t_from_1dim_array(2, values[2])),
        t_lock(t_from_1dim_array(2, values[3])),
    };

    tensor *t_truths[4] = {
        t_lock(t_from_1dim_array(2, truths[0])),
        t_lock(t_from_1dim_array(2, truths[1])),
        t_lock(t_from_1dim_array(2, truths[2])),
        t_lock(t_from_1dim_array(2, truths[3])),
    };

    struct seqmodel *s = seqmodel_init();
    seqmodel_push(s, input_layer_init(2));
    seqmodel_push(s, dense_layer_init(2, 2, av_tanh));
    seqmodel_push(s, dense_layer_init(2, 2, av_softmax));

    for (int i = 0; i < 4; i++)
    {
        tensor *t = seqmodel_predict(s, t_values[i]);
        t_print(t);
        t_free(t);
    }

    struct optimizer *o = opt_adam_init(0.01, 0.9, 0.999, loss_mse);
    // struct optimizer *o = opt_sgd_init(0.1, 0.9, loss_mse);
    train(s, 4, t_values, t_truths, MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    for (int i = 0; i < 4; i++)
    {
        t_print(seqmodel_predict(s, t_values[i]));
    }
}

void seq_example_XOR()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    seq_example_EXEC(values, truths);
}

void seq_example_OR()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    seq_example_EXEC(values, truths);
}

void seq_example_AND()
{
    param_t values[4][2] = {{0.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 0.0},
                            {1.0, 1.0}};
    param_t truths[4][2] = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
    seq_example_EXEC(values, truths);
}