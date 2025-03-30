#include "ocamlrt.h"
#include <cstdarg>
#include <cstdio>

void *ocamlrt_print_float(void *p) {
    printf("%f\n", *(double *)p);
    return p;
}

void *ocamlrt_box_convert_i64_f64(void *p) {
    auto *f64 = new double(*(int64_t *)p);
    return (void *)f64;
}

void *ocamlrt_box_convert_f64_i64(void *p) {
    auto *i64 = new int64_t(*(double *)p);
    return (void *)i64;
}

void *ocamlrt_embox_i64(int64_t i64) {
    auto *p = new int64_t(i64);
    return (void *)p;
}
