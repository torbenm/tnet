#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "examples.h"
#include "core.h"
#include "train.h"

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
    printf("No such function: %s. Valid values:\n", funcName);
    for (int i = 0; i < numFuncs; i++)
    {
        printf("- %s\n", clifuncs[i]->name);
    }
    return 1;
}

int main(int argc, char **argv)
{
    const int numFuncs = 7;
    struct clifunc *funcs[numFuncs] = {
        clifunc_init("perceptron-or", perceptron_example_OR),
        clifunc_init("perceptron-and", perceptron_example_AND),
        clifunc_init("perceptron-xor", perceptron_example_XOR),
        clifunc_init("seq-xor", seq_example_XOR),
        clifunc_init("seq-and", seq_example_AND),
        clifunc_init("seq-or", seq_example_OR),
        clifunc_init("train-test", train_test),   // bit hacky way to include testing...
        clifunc_init("tensor-test", tensor_test), // bit hacky way to include testing...
    };

    tnet_init();
    if (argc == 2)
    {
        return clifunc_iterexec(numFuncs, funcs, argv[1]);
    }
    else
    {
        printf("Unknown number of args: %i, only one expected. Valid values:\n", argc - 1);
        for (int i = 0; i < numFuncs; i++)
        {
            printf("- %s\n", funcs[i]->name);
        }
        return 1;
    }
    return 0;
}