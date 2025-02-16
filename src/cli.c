#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <argp.h>
#include <errno.h>

#include "cli.h"
#include "core.h"

const char *argp_program_version = "tnet 0.1";

/* The options for the 'train' subcommand. */
static struct argp_option train_options[] = {
    {"model", 'm', "STRING", 0, "Path to the model definition file."},
    {"data", 'd', "STRING", 0, "Path to the data file."},
    {"target-column", 't', "NUMBER", 0, "Index of the target column."},
    {"opt", 'o', "STRING", 0, "Optimizer ('adam' or 'sgd')"},
    {0}};

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    arguments *arguments = state->input;

    switch (key)
    {
    case 'm':
        arguments->model_file = arg;
        break;
    case 'd':
        arguments->data_file = arg;
        break;
    case 't':
        arguments->target_column = (int)strtol(arg, (char **)NULL, 10);
        if (errno > 0)
            error("Failed to parse '%s' - %s.", arg, strerror(errno));
        break;
    case 'o':
        if (strcmp(arg, "adam") == 0)
            arguments->opt = OPT_ADAM;
        else if (strcmp(arg, "sgd") == 0)
            arguments->opt = OPT_SGD;
        else
            argp_error(state, "Invalid value for --opt: %s", arg);
        break;
    case ARGP_KEY_ARG:
        if (state->arg_num == 0)
        {
            arguments->subcommand = arg;
        }
        else
        {
            argp_usage(state);
        }
        break;
    case ARGP_KEY_END:
        if (state->arg_num < 1)
        {
            argp_usage(state);
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

/* Subcommand parsers. */
static struct argp train_argp = {train_options, parse_opt, "[train|perceptron] [OPTIONS]", "tnet let's you train and run models."};

/* Main entry point. */
int main(int argc, char **argv)
{
    tnet_init();

    arguments arguments;
    memset(&arguments, 0, sizeof(arguments));

    /* Parse the arguments. */
    argp_parse(&train_argp, argc, argv, ARGP_IN_ORDER, 0, &arguments);

    /* Execute the chosen subcommand. */
    if (strcmp(arguments.subcommand, "train") == 0)
        command_train(&arguments);
    else if (strcmp(arguments.subcommand, "check") == 0)
        command_check(&arguments);
    else if (strcmp(arguments.subcommand, "predict") == 0)
        command_predict(&arguments);
    else
    {
        fprintf(stderr, "Unknown subcommand: %s\n", arguments.subcommand);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}