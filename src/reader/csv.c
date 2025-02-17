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
    if (c == NULL)
        error("Failed to allocate csv_reader");
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
char *csv_next_field(csv_reader *c, int *outReadStatus)
{
    char buffer[BUFFER_SIZE];
    int buf_ptr;
    int ch;
    int isEscaped = 0;
    if (outReadStatus != NULL)
        *outReadStatus = READ_STATUS_MORE;
    if (c == NULL || c->file == NULL)
        error("Invalid file pointer.");

    for (buf_ptr = 0; buf_ptr < BUFFER_SIZE;)
    {
        ch = fgetc(c->file);

        if (ch == '\n')
        {
            if (outReadStatus != NULL)
                *outReadStatus = READ_STATUS_EOL;
            break;
        }
        if (ch == EOF)
        {
            if (outReadStatus != NULL)
                *outReadStatus = READ_STATUS_EOF;
            break;
        }
        if (ch == ESCAPE)
        {
            isEscaped = !isEscaped;
            continue;
        }
        else if (ch == DELIMITER && !isEscaped)
            break; // skipping to end of loop
        buffer[buf_ptr++] = (char)ch;
    }
    if (buf_ptr == 0)
    {
        return NULL;
    }
    buffer[buf_ptr++] = '\0';
    char *r = mm_alloc(sizeof(char) * buf_ptr);

    strncpy(r, buffer, buf_ptr);
    return r;
}

/**
 * Reads the next field (until ';') and returns it as a parsed param
 */
param_t str_to_param(const char *field)
{
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
int csv_next_field_int(csv_reader *c, int *outReadStatus)
{
    char *field = csv_next_field(c, outReadStatus);
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
 * You may pass in a readStatus from a previous csv_next_field call here.
 * It will behave such as:
 * - READ_STATUS_MORE -> read until new line character or EOF
 * - READ_STATUS_EOL -> immediately return READ_STATUS_EOL
 * - READ_STATUS_EOF -> immediately return READ_STATUS_EOF
 */
int csv_seek_next_line(csv_reader *c, int readStatus)
{
    if (readStatus > READ_STATUS_MORE)
        return readStatus;
    int ch;
    if (c == NULL || c->file == NULL)
        error("No valid csv_reader object provided.");
    while ((ch = fgetc(c->file)) != EOF)
    {
        if (ch == '\n') // new line found
            return READ_STATUS_EOL;
    }
    return READ_STATUS_EOF;
}