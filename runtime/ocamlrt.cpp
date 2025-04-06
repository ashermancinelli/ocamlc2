#include "OCamlRT.h"
#include <cassert>
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

static void debugI64(Value v) {
    DBGS("i64: %p %lld\n", v.p, v.components.rest);
}

static void debugF64(Value v) {
    DBGS("f64: %p %f\n", v.p, *(double *)v.p);
}

static void debugString(Value v) {
    DBGS("string: %p %s\n", v.p, (char *)v.p);
}

static void debugVariant(Value v) {
    Variant *variantPointer = (Variant *)v.p;
    DBGS("variant: %p %lld\n", v.p, unbox_i64(variantPointer->kind));
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
    debugI64(lhs);
    debugI64(rhs);
    int64_t result = lhs.components.rest + rhs.components.rest;
    return embox_i64(result);
}

Value sub_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    debugI64(lhs);
    debugI64(rhs);
    int64_t result = lhs.components.rest - rhs.components.rest;
    return embox_i64(result);
}

Value mul_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    debugI64(lhs);
    debugI64(rhs);
    int64_t result = lhs.components.rest * rhs.components.rest;
    return embox_i64(result);
}

Value div_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    debugI64(lhs);
    debugI64(rhs);
    int64_t result = lhs.components.rest / rhs.components.rest;
    return embox_i64(result);
}

Value mod_i64_i64(Value lhs, Value rhs) {
    DBGS("\n");
    debugI64(lhs);
    debugI64(rhs);
    int64_t result = lhs.components.rest % rhs.components.rest;
    return embox_i64(result);
}

Value print_float(Value p) {
    DBGS("\n");
    debugF64(p);
    printf("%f\n", *(double *)p.p);
    return unit();
}

Value print_int(Value p) {
    DBGS("\n");
    debugI64(p);
    printf("%lld\n", (int64_t)p.components.rest);
    return unit();
}

Value box_convert_i64_f64(Value p) {
    DBGS("\n");
    debugI64(p);
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
    debugI64(v);
    return v;
}

Value embox_i64(int64_t i64) {
    DBGS("%lld\n", i64);
    Value v;
    v.components.tag = 1;
    v.components.rest = i64;
    debugI64(v);
    return v;
}

int64_t unbox_i64(Value v) {
    return v.components.rest;
}

Value embox_f64(double f64) {
    DBGS("%f\n", f64);
    auto *p = ocaml_malloc<double>();
    *p = f64;
    Value v;
    v.p = p;
    debugF64(v);
    return v;
}

double unbox_f64(Value v) {
    return *(double *)v.p;
}

Value embox_string(const char *str) {
    DBGS("%s\n", str);
    std::string s(str);
    auto *p = ocaml_malloc<char>(s.size());
    std::memcpy(p, s.data(), s.size());
    Value v;
    v.p = p;
    debugString(v);
    return v;
}

const char *unbox_string(Value v) {
    return (char *)v.p;
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

Value variant_ctor_empty(Value activeVariant) {
    DBGS("\n");
    auto *p = ocaml_malloc<Variant>();
    p->kind = activeVariant;
    p->data = Value{.p = (void *)0xdeadbeef};
    Value v;
    v.p = p;
    debugVariant(v);
    return v;
}

Value variant_ctor(Value activeVariant, Value data) {
    DBGS("\n");
    auto *p = ocaml_malloc<Variant>();
    p->kind = activeVariant;
    p->data = data;
    Value v;
    v.p = p;
    debugVariant(v);
    return v;
}

Value variant_get_kind(Value v) {
    DBGS("\n");
    debugVariant(v);
    Variant *variantPointer = (Variant *)v.p;
    return variantPointer->kind;
}

Value variant_get_data(Value v) {
    DBGS("\n");
    debugVariant(v);
    Variant *variantPointer = (Variant *)v.p;
    return variantPointer->data;
}

Value tuple_ctor(Value numElementsValue, ...) {
    DBGS("\n");
    va_list args;
    va_start(args, numElementsValue);
    auto numElements = unbox_i64(numElementsValue);
    auto *tuple = ocaml_malloc<Tuple>();
    tuple->numElements = numElementsValue;
    tuple->elements = ocaml_malloc<Value>(numElements);
    for (int i = 0; i < numElements; i++) {
        Value v = va_arg(args, Value);
        tuple->elements[i] = v;
    }
    Value v;
    v.p = tuple;
    return v;
}

Value tuple_get(Value tuple, Value index) {
    DBGS("\n");
    Tuple *tuplePointer = (Tuple *)tuple.p;
    auto indexValue = unbox_i64(index);
    auto numElements = unbox_i64(tuplePointer->numElements);
    assert(indexValue < numElements && "Attempted out-of-bounds tuple access");
    return tuplePointer->elements[indexValue];
}
