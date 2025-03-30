#include "ocamlrt.h"
#include <cstdarg>
#include <cstdio>

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
    auto *f64 = new double(*(int64_t *)p.p);
    Value v;
    v.p = f64;
    return v;
}

Value box_convert_f64_i64(Value p) {
    auto *i64 = new int64_t(*(double *)p.p);
    Value v;
    v.p = i64;
    return v;
}

Value embox_i64(int64_t i64) {
    auto *p = new int64_t(i64);
    Value v;
    v.p = p;
    return v;
}

Value embox_f64(double f64) {
    auto *p = new double(f64);
    Value v;
    v.p = p;
    return v;
}
