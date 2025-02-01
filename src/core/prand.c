#include <stdlib.h>

#include "core.h"

param_t prand()
{
    return (param_t)rand() / ((param_t)RAND_MAX);
}