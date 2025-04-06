#pragma once
#include <cstdint>
#define OCAML_PTR_ALIGNMENT 8
extern "C" {
enum {
  OCAML_TAG_INT = 5,
};
struct ValueComponents {
  int tag : 3;
  int64_t rest : 61;
};
struct Value {
  union {
    ValueComponents components;
    int64_t i;
    void *p;
  };
};

struct Tuple {
  Value numElements;
  Value *elements;
};

struct Variant {
    Value kind;
    Value data;
};

#define RT_PREFIX "_ocamlrt."
#define DECLARE_RT(NAME, RETURN_TYPE, ...) \
  RETURN_TYPE NAME(__VA_ARGS__) __asm__(RT_PREFIX #NAME);
#define DECLARE_RT_AS(NAME, ASM_NAME, RETURN_TYPE, ...) \
  RETURN_TYPE NAME(__VA_ARGS__) __asm__(RT_PREFIX ASM_NAME);

DECLARE_RT(unit, void, Value);
DECLARE_RT(print_float, Value, Value p);
DECLARE_RT(print_int, Value, Value p);
DECLARE_RT(mul_i64_i64, Value, Value lhs, Value rhs);
DECLARE_RT(add_i64_i64, Value, Value lhs, Value rhs);
DECLARE_RT(sub_i64_i64, Value, Value lhs, Value rhs);
DECLARE_RT(div_i64_i64, Value, Value lhs, Value rhs);
DECLARE_RT(mod_i64_i64, Value, Value lhs, Value rhs);
DECLARE_RT(box_convert_i64_f64, Value, Value p);
DECLARE_RT(box_convert_f64_i64, Value, Value p);
DECLARE_RT(embox_i64, Value, int64_t i64);
DECLARE_RT(unbox_i64, int64_t, Value v);
DECLARE_RT(embox_f64, Value, double f64);
DECLARE_RT(unbox_f64, double, Value v);
DECLARE_RT(embox_string, Value, const char *str);
DECLARE_RT(unbox_string, const char *, Value v);
DECLARE_RT_AS(ocaml_printf, "Printf.printf", Value, const char *format, ...);
DECLARE_RT(variant_ctor_empty, Value, Value activeVariant);
DECLARE_RT(variant_ctor, Value, Value activeVariant, Value data);
DECLARE_RT(variant_get_kind, Value, Value v);
DECLARE_RT(variant_get_data, Value, Value v);
DECLARE_RT(tuple_ctor, Value, Value numElements);
DECLARE_RT(tuple_get, Value, Value tuple, Value index);
}
