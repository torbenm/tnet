#include "cli.h"
#include "reader.h"
#include "core.h"
#include "models.h"
#include "train.h"

// Default parameters - might be configurable via argp at some point as well.
#define COST_THRESHOLD 0.0001
#define DIFF_THRESHOLD 0.000000001
#define MAX_ITER 10000000
#define ALPHA_SGD 0.1
#define ALPHA_ADAM 0.01
#define BETA1 0.9
#define BETA2 0.999
#define LOSS_FN loss_binary_cross_entropy()

struct optimizer *__get_opt_from_arg(arguments *args)
{
    switch (args->opt)
    {
    case OPT_ADAM:
        return opt_adam_init(ALPHA_ADAM, BETA1, BETA2, LOSS_FN);
    default:
        return opt_sgd_init(ALPHA_SGD, 0, LOSS_FN);
    }
}

seqmodel *__load_model_and_data(arguments *args, tensor ***outInputs, tensor ***outTruths, int *outBatchSize)
{
    print_header("Loading models & inputs from file");

    data_header *header = mm_alloc(sizeof(data_header));
    *outBatchSize = parse_csv_into_inputs_and_truth(args->data_file, outInputs, outTruths, args->target_column, header);

    printf("Read %i rows from data file.", *outBatchSize);
    print_data_header(header);

    seqmodel *s = seqmodel_from_file(args->model_file);
    seqmodel_print(s);
    return s;
}

void command_train(arguments *args)
{
    tensor **inputs;
    tensor **truths;
    int batchSize;
    seqmodel *s = __load_model_and_data(args, &inputs, &truths, &batchSize);

    optimizer *o = __get_opt_from_arg(args);

    print_header("Initial predictions.");
    for (int i = 0; i < batchSize; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
    }
    print_header("Starting optimization process with %s.", o->name);
    train(s, batchSize, inputs, truths, MAX_ITER, o, DIFF_THRESHOLD, COST_THRESHOLD);

    print_header("Calculating final predictions.");
    for (int i = 0; i < batchSize; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
    }
}

void command_predict(arguments *args)
{
    tensor **inputs;
    tensor **truths;
    int batchSize;
    seqmodel *s = __load_model_and_data(args, &inputs, &truths, &batchSize);

    print_header("Predicting.");
    for (int i = 0; i < batchSize; i++)
    {
        t_print(seqmodel_predict(s, inputs[i]));
        printf("\n");
    }
}

void command_check(arguments *args)
{
    tensor **inputs;
    tensor **truths;
    int batchSize;
    seqmodel *s = __load_model_and_data(args, &inputs, &truths, &batchSize);

    print_header("Checking gradients.");
    check_gradients(s, batchSize, inputs, truths, LOSS_FN);
}
