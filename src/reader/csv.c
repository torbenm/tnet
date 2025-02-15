#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "core.h"
#include "reader.h"

#define BUFFER_SIZE 1024
#define DELIMITER ';'
#define ESCAPE '"'

/**
 * Opens a new file and then skips the header line.
 */
csv_reader *csv_open(const char *filename)
{
    csv_reader *c = mm_calloc(1, sizeof(csv_reader));
    c->file = fopen(filename, "r");
    if (c->file == NULL)
        error("Error opening file %s!", filename);
    return c;
}

void csv_close(csv_reader *c)
{
    fclose(c->file);
    mm_free(c);
}

/**
 * Reads the next field as 'string'. May at most be 1024 characters long
 */
char *csv_next_field(csv_reader *c)
{
    char buffer[BUFFER_SIZE];
    int buf_ptr;
    char ch;
    int isEscaped = 0;
    for (buf_ptr = 0; buf_ptr < BUFFER_SIZE; buf_ptr++)
    {
        ch = fgetc(c->file);
        if (ch == EOF || ch == '\n')
        {
            // going one char back in the file to allow 'csv_seek_next_line'
            // to be called
            fseek(c->file, -1, SEEK_CUR);
            break;
        }
        if (ch == ESCAPE)
        {
            isEscaped = !isEscaped;
            continue;
        }
        else if (ch == DELIMITER && !isEscaped)
            break; // skipping to end of loop
        buffer[buf_ptr] = ch;
    }
    if (buf_ptr == 0)
        return NULL;
    buffer[buf_ptr++] = '\0';
    char *r = mm_alloc(sizeof(char) * buf_ptr);

    strncpy(r, buffer, buf_ptr);
    return r;
}

/**
 * Reads the next field (until ';') and returns it as a parsed param
 */
param_t csv_next_field_param(csv_reader *c)
{
    char *field = csv_next_field(c);
    if (field == NULL)
        error("Could not read non-nullable field that is null as param.");
    int result = (param_t)strtod(field, (char **)NULL);
    if (result == 0 && errno > 0)
    {
        error("Failed to parse param from %s - '%s'.", field, strerror(errno));
    }
    return result;
}

/**
 * Reads the next field (until ';') and returns it as a parsed int
 */
int csv_next_field_int(csv_reader *c)
{
    char *field = csv_next_field(c);
    if (field == NULL)
        error("Could not read non-nullable field that is null as integer.");
    int result = (int)strtol(field, (char **)NULL, 10);
    if (result == 0 && errno > 0)
    {
        error("Failed to parse integer from %s - '%s'.", field, strerror(errno));
    }
    return result;
}

/**
 * Continues on the buffer until it finds a '\n'.
 * Returns EOF if the end of file is reached
 */
int csv_seek_next_line(csv_reader *c)
{
    char ch;
    while ((ch = fgetc(c->file)) != EOF)
    {
        if (ch == '\n') // new line found
            return 0;
    }
    return EOF;
}