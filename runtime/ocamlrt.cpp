#include "OCamlRT.h"
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <gc.h>
#include "OCamlRTConfig.h"
#include "OCamlRTDebug.h"

template <typename T>
T *ocaml_malloc(size_t size=1) {
    return (T *)GC_memalign(OCAML_PTR_ALIGNMENT, size * sizeof(T));
}

Value unit() {
    DBGS("\n");
    Value v;
    v.i = 0;
    v.components.tag = 1;
    DBGS("unit result: %p\n", v.p);
    return v;
}

Value add_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    int64_t result = lhs.components.rest + rhs.components.rest;
    return embox_i64(result);
}

Value sub_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    int64_t result = lhs.components.rest - rhs.components.rest;
    return embox_i64(result);
}

Value mul_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    int64_t result = lhs.components.rest * rhs.components.rest;
    return embox_i64(result);
}

Value div_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    int64_t result = lhs.components.rest / rhs.components.rest;
    return embox_i64(result);
}

Value mod_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    int64_t result = lhs.components.rest % rhs.components.rest;
    return embox_i64(result);
}

Value print_float(Value p) {
    DBGS("\n");
    printf("%f\n", *(double *)p.p);
    return unit();
}

Value print_int(Value p) {
    DBGS("\n");
    printf("%lld\n", (int64_t)p.components.rest);
    return unit();
}

Value box_convert_i64_f64(Value p) {
    DBGS("%p\n", p.p);
    auto *f64 = ocaml_malloc<double>();
    *f64 = (double)(int64_t)p.components.rest;
    Value v;
    v.p = f64;
    DBGS("%p %f\n", v.p, *f64);
    return v;
}

Value box_convert_f64_i64(Value p) {
    DBGS("%p\n", p.p);
    auto *i64 = ocaml_malloc<int64_t>();
    *i64 = *(double *)p.p;
    Value v;
    v.p = i64;
    return v;
}

Value embox_i64(int64_t i64) {
    DBGS("%lld\n", i64);
    Value v;
    v.components.tag = 1;
    v.components.rest = i64;
    DBGS("%p\n", v.p);
    return v;
}

Value embox_f64(double f64) {
    DBGS("%f\n", f64);
    auto *p = ocaml_malloc<double>();
    *p = f64;
    Value v;
    v.p = p;
    DBGS("%p\n", v.p);
    return v;
}

Value embox_string(const char *str) {
    DBGS("%s\n", str);
    std::string s(str);
    auto *p = ocaml_malloc<char>(s.size());
    std::memcpy(p, s.data(), s.size());
    Value v;
    v.p = p;
    DBGS("%p\n", v.p);
    return v;
}

Value ocaml_printf(const char *format, ...) {
    DBGS("\n");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    DBGS("\n");
    return unit();
}
