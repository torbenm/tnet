#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void tnet_init()
{
    srand(time(NULL));
    // #if DEBUG
    setbuf(stdout, 0);
    // #endif
}