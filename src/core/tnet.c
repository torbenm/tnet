#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void tnet_init()
{
    // srand(100);
    srand(time(NULL));
    // srand(35);
    setbuf(stdout, 0);
}