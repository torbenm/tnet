#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void tnet_init()
{
    srand(time(NULL));
    rand();
    setbuf(stdout, 0);
}