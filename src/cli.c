#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "perceptron_ex.h"
#include "mlp_ex.h"
#include "tnet.h"

struct clifunc
{
    const char *name;
    void (*func)(void);
};

struct clifunc *clifunc_init(const char *name, void (*func)(void))
{
    struct clifunc *cf = calloc(1, sizeof(struct clifunc));
    cf->name = name;
    cf->func = func;
    return cf;
}

int clifunc_iterexec(int numFuncs, struct clifunc *clifuncs[numFuncs], char *funcName)
{
    for (int i = 0; i < numFuncs; i++)
    {
        if (strcmp(funcName, clifuncs[i]->name) == 0)
        {
            clifuncs[i]->func();
            return 0;
        }
    }
    printf("No such function: %s. Valid values: [", funcName);
    for (int i = 0; i < numFuncs; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%s", clifuncs[i]->name);
    }
    printf("]\n");
    return 1;
}

int cli_example(int argc, char **argv, int numFuncs, struct clifuncs *funcs[numFuncs])
{
    if (argc == 3)
    {
        return clifunc_iterexec(numFuncs, funcs, argv[2]);
    }
    else
    {
        printf("Expected one more argument for which program to run, found %i arguments.\n", argc - 2);
        return 1;
    }
}

int main(int argc, char **argv)
{
    tnet_init();
    if (argc >= 2)
    {
        if (strcmp(argv[1], "perceptron") == 0)
        {
            const int numFuncs = 3;
            struct clifunc *funcs[numFuncs] = {
                clifunc_init("or", perceptron_example_OR),
                clifunc_init("and", perceptron_example_AND),
                clifunc_init("xor", perceptron_example_XOR)};
            return cli_example(argc, argv, numFuncs, funcs);
        }
        else if (strcmp(argv[1], "mlp") == 0)
        {
            const int numFuncs = 1;
            struct clifunc *funcs[numFuncs] = {
                clifunc_init("xor", mlp_example_XOR)};
            return cli_example(argc, argv, numFuncs, funcs);
        }

        else
        {
            printf("Unknown command: %s\n", argv[1]);
            return 1;
        }
    }
    else
    {
        printf("Unknown number of args: %i, more than 2\n", argc);
        return 1;
    }
    return 0;
}