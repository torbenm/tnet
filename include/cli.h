

#define OPT_SGD 1
#define OPT_ADAM 2

/* Used by main to communicate with parse_opt. */
typedef struct
{
    char *subcommand;
    char *model_file;
    char *data_file;
    int target_column;
    int opt;
} arguments;

void command_train(arguments *args);
void command_check(arguments *args);
void command_predict(arguments *args);
