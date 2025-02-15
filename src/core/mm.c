#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "core.h"

/**
 * "Basic" Garbage collector & memory management.
 * All allocated memory is stored in a list of pointers.
 * You may run mm_sweep(); every now and then - this will go through the list of pointers
 * and remove all memory allocations that have not been markd before.
 */

/**
 * Defines the maximum number of pointers - currently hardcoded
 */
#define MAX_MEMORY_POINTERS 4096

void *MEMORY_POINTERS[MAX_MEMORY_POINTERS];
void *MARKED_MM_PTRS[MAX_MEMORY_POINTERS];
size_t next_memory_ptr = 0;
size_t next_markd_ptr = 0;

void __mm_compact()
{
    // compact the pointers list
    void *new_buffer[MAX_MEMORY_POINTERS];
    size_t new_buffer_ptr = 0;
    for (int i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] != NULL)
            new_buffer[new_buffer_ptr++] = MEMORY_POINTERS[i];
    }

    // just changing the pointers to this probably is more effective,
    // but this is easier right now...
    next_memory_ptr = 0;
    for (int i = 0; i < new_buffer_ptr; i++)
    {
        MEMORY_POINTERS[next_memory_ptr++] = new_buffer[i];
    }
}

void __mm_compact_if_needed()
{
    if (next_memory_ptr > MAX_MEMORY_POINTERS)
    {
        __mm_compact();
        if (next_memory_ptr > MAX_MEMORY_POINTERS) // let's see if this fixed it...
            error("You tried to allocate too much memory :(. Either try wiping more often or make the memory ptrs list more flexible...\n");
    }
}

void *mm_alloc(size_t __size)
{
    __mm_compact_if_needed();
    MEMORY_POINTERS[next_memory_ptr] = malloc(__size);
    if (MEMORY_POINTERS[next_memory_ptr] == NULL)
    {
        error("Failed to allocate memory with malloc.");
    }
    return MEMORY_POINTERS[next_memory_ptr++];
}
void *mm_calloc(size_t __count, size_t __size)
{
    __mm_compact_if_needed();

    MEMORY_POINTERS[next_memory_ptr] = calloc(__count, __size);
    if (MEMORY_POINTERS[next_memory_ptr] == NULL)
    {
        error("Failed to allocate memory with calloc.");
    }
    return MEMORY_POINTERS[next_memory_ptr++];
}
void mm_free(void *ptr)
{
    for (int i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] == ptr)
        {
            MEMORY_POINTERS[i] = NULL;
            free(ptr);
        }
    }
}
void mm_mark(void *ptr)
{
    if (next_markd_ptr > MAX_MEMORY_POINTERS)
        error("You tried to mark too many pointers :(. Either try wiping more often or make the memory ptrs list more flexible...\n");
    MARKED_MM_PTRS[next_markd_ptr++] = ptr;
}
void mm_unmark(void *ptr)
{
    for (int i = 0; i < next_markd_ptr; i++)
    {
        if (MARKED_MM_PTRS[i] == ptr)
        {
            MARKED_MM_PTRS[i] = NULL;
        }
    }
}
void mm_unmark_all()
{
    // just setting next_markd_ptr to 0
    next_markd_ptr = 0;
}

int __mm_sweep_ptr(int memoryPtrIdx)
{
    for (int m = 0; m < next_markd_ptr; m++)
    {
        if (MARKED_MM_PTRS[m] == MEMORY_POINTERS[memoryPtrIdx])
            return 0;
    }
    free(MEMORY_POINTERS[memoryPtrIdx]);
    MEMORY_POINTERS[memoryPtrIdx] = NULL;
    return 1;
}

void mm_sweep()
{
    for (int i = 0; i < next_memory_ptr; i++)
    {
        if (MEMORY_POINTERS[i] != NULL)
            __mm_sweep_ptr(i);
    }
    __mm_compact();
}