#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>

#include "core.h"

param_t prand()
{
    return (param_t)rand() / RAND_MAX * 2.0 - 1.0;
}

void error(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    pthread_exit(NULL);
}

param_t clip(param_t input, param_t lowerBound, param_t upperBound)
{
    if (input < lowerBound)
        return lowerBound;
    if (input > upperBound)
        return upperBound;
    return input;
}