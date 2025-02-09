#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void tnet_init()
{
    srand(time(NULL));
    setbuf(stdout, 0);
}