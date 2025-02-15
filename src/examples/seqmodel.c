#include <stdio.h>

#include "core.h"
#include "models.h"
#include "funcs.h"
#include "train.h"
#include "reader.h"

#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 10000000
#define LEARNING_RATE 0.1
#define MONUMENTUM 0.9

void seq_example_EXEC(const char *inputsFile, const char *truthsFile)
{
    int numInputs, numTruths;
    tensor **inputs = tensor_array_from_file(inputsFile, &numInputs);
    tensor **truths = tensor_array_from_file(truthsFile, &numTruths);

    if (numInputs != numTruths)
        error("Can only train when number of inputs equals number of truths - got %i != %i.", numInputs, numTruths);

    struct seqmodel *s = seqmodel_from_file("./data/binary_ops/model.csv");
    seqmodel_print(s);

    for (int i = 0; i < 4; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
    }
    struct optimizer *o = opt_sgd_init(0.1, 0, loss_mse);
    // struct optimizer *o = opt_adam_init(0.1, 0.9, 0.999, loss_mse);

    train(s, numInputs, inputs, truths, MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    for (int i = 0; i < 4; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
    }
}

void seq_example_XOR()
{
    seq_example_EXEC("data/binary_ops/inputs.csv", "data/binary_ops/truth_xor.csv");
}

void seq_example_OR()
{
    seq_example_EXEC("data/binary_ops/inputs.csv", "./data/binary_ops/truth_or.csv");
}

void seq_example_AND()
{
    seq_example_EXEC("data/binary_ops/inputs.csv", "./data/binary_ops/truth_and.csv");
}