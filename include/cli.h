

#define OPT_SGD 1
#define OPT_ADAM 2

/* Used by main to communicate with parse_opt. */
struct arguments
{
    char *subcommand;
    char *model;
    char *inputs;
    char *truths;
    int opt;
};

void command_train(struct arguments *args);
void command_check(struct arguments *args);
