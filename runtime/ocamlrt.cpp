#include "ocamlrt.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include <gc.h>

template <typename T>
T *ocaml_malloc(size_t size=1) {
    return (T *)GC_memalign(OCAML_PTR_ALIGNMENT, size * sizeof(T));
}

Value unit() {
    Value v;
    v.i = 0;
    v.components.tag = 1;
    return v;
}

Value print_float(Value p) {
    printf("%f\n", *(double *)p.p);
    return unit();
}

Value box_convert_i64_f64(Value p) {
    auto *f64 = ocaml_malloc<double>();
    *f64 = *(int64_t *)p.p;
    Value v;
    v.p = f64;
    return v;
}

Value box_convert_f64_i64(Value p) {
    auto *i64 = ocaml_malloc<int64_t>();
    *i64 = *(double *)p.p;
    Value v;
    v.p = i64;
    return v;
}

Value embox_i64(int64_t i64) {
    auto *p = ocaml_malloc<int64_t>();
    *p = i64;
    Value v;
    v.p = p;
    return v;
}

Value embox_f64(double f64) {
    auto *p = ocaml_malloc<double>();
    *p = f64;
    Value v;
    v.p = p;
    return v;
}

Value embox_string(const char *str) {
    std::string s(str);
    auto *p = ocaml_malloc<char>(s.size());
    std::memcpy(p, s.data(), s.size());
    Value v;
    v.p = p;
    return v;
}

Value ocaml_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    // vprintf(format, args);
    puts("Called printf\n");
    return unit();
}
