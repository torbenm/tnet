#include "cli.h"
#include "reader.h"
#include "core.h"
#include "models.h"
#include "train.h"

// Default parameters - might be configurable via argp at some point as well.
#define COST_THRESHOLD 0.001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 10000000
#define ALPHA_SGD 0.1
#define ALPHA_ADAM 0.01
#define BETA1 0.9
#define BETA2 0.999
#define LOSS_FN loss_mse

struct optimizer *__get_opt_from_arg(struct arguments *args)
{
    switch (args->opt)
    {
    case OPT_ADAM:
        return opt_adam_init(ALPHA_ADAM, BETA1, BETA2, LOSS_FN);
    default:
        return opt_sgd_init(ALPHA_SGD, 0, LOSS_FN);
    }
}

void command_train(struct arguments *args)
{
    print_header("Loading models & inputs from file");
    int numInputs, numTruths;
    tensor **inputs = tensor_array_from_file(args->inputs, &numInputs);
    tensor **truths = tensor_array_from_file(args->truths, &numTruths);

    if (numInputs != numTruths)
        error("Can only train when number of inputs equals number of truths - got %i != %i.", numInputs, numTruths);

    struct seqmodel *s = seqmodel_from_file(args->model);
    seqmodel_print(s);

    print_header("Starting optimization process.");
    struct optimizer *o = __get_opt_from_arg(args);
    train(s, numInputs, inputs, truths, MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    print_header("Calculating final predictions.");
    for (int i = 0; i < numInputs; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
    }
}
