#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define COLOR_HEADER "\033[1;38;5;2m"
#define COLOR_OFF "\e[m"

void __cvprintf(const char *color_code, const char *msg, va_list args)
{
    printf("%s", color_code);
    vprintf(msg, args);
    printf(COLOR_OFF);
}

void __cprintf(const char *color_code, const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    __cvprintf(color_code, msg, args);
    va_end(args);
}

void print_header(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    __cprintf(COLOR_HEADER, ">> ");
    __cvprintf(COLOR_HEADER, msg, args);
    printf("\n");
    va_end(args);
}

void print_int_array(int *a, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        if (i > 0)
            printf(", ");
        printf("%i", a[i]);
    }
    printf("]");
}
