#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "core.h"

#define MAX_MEMORY_POINTERS 4096

static void *MEMORY_POINTERS[MAX_MEMORY_POINTERS] = {NULL};
static void *MARKED_MM_PTRS[MAX_MEMORY_POINTERS] = {NULL};
static size_t next_memory_ptr = 0;
static size_t next_markd_ptr = 0;

void __mm_compact()
{
    size_t new_buffer_ptr = 0;
    for (size_t i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] != NULL)
        {
            MEMORY_POINTERS[new_buffer_ptr++] = MEMORY_POINTERS[i];
        }
    }
    next_memory_ptr = new_buffer_ptr;
}

void __mm_compact_if_needed()
{
    if (next_memory_ptr >= MAX_MEMORY_POINTERS)
    {
        __mm_compact();
        if (next_memory_ptr >= MAX_MEMORY_POINTERS)
        {
            error("You tried to allocate too much memory. Consider increasing MAX_MEMORY_POINTERS.");
        }
    }
}

void *mm_alloc(size_t __size)
{
    __mm_compact_if_needed();
    void *ptr = malloc(__size);
    if (ptr == NULL)
    {
        error("Failed to allocate memory with malloc.");
    }
    MEMORY_POINTERS[next_memory_ptr++] = ptr;
    return ptr;
}

void *mm_calloc(size_t __count, size_t __size)
{
    __mm_compact_if_needed();
    void *ptr = calloc(__count, __size);
    if (ptr == NULL)
    {
        error("Failed to allocate memory with calloc.");
    }
    MEMORY_POINTERS[next_memory_ptr++] = ptr;
    return ptr;
}

void mm_free(void *ptr)
{
    if (ptr == NULL)
        return;
    for (size_t i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] == ptr)
        {
            free(ptr);
            MEMORY_POINTERS[i] = NULL;
            break;
        }
    }
}

void mm_mark(void *ptr)
{
    if (next_markd_ptr >= MAX_MEMORY_POINTERS)
    {
        error("You tried to mark too many pointers. Consider increasing MAX_MEMORY_POINTERS.");
    }
    MARKED_MM_PTRS[next_markd_ptr++] = ptr;
}

void mm_unmark(void *ptr)
{
    for (size_t i = 0; i < next_markd_ptr; i++)
    {
        if (MARKED_MM_PTRS[i] == ptr)
        {
            MARKED_MM_PTRS[i] = NULL;
            break;
        }
    }
}

void mm_unmark_all()
{
    next_markd_ptr = 0;
}

bool __mm_is_marked(void *ptr)
{
    for (size_t m = 0; m < next_markd_ptr; m++)
    {
        if (MARKED_MM_PTRS[m] == ptr)
        {
            return true;
        }
    }
    return false;
}

void mm_sweep()
{
    for (size_t i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] != NULL && !__mm_is_marked(MEMORY_POINTERS[i]))
        {
            free(MEMORY_POINTERS[i]);
            MEMORY_POINTERS[i] = NULL;
        }
    }
    __mm_compact();
}