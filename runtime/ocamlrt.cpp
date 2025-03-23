#include "ocamlrt.h"
#include <cstdarg>
#include <cstdio>

void ocamlrt_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
