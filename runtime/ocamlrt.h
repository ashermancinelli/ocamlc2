#pragma once
#include <cstdint>
#define OCAML_PTR_ALIGNMENT 8
extern "C" {
enum {
  OCAML_TAG_INT = 5,
};
struct ValueComponents {
  int tag : 3;
  unsigned long rest : 61;
};
struct Value {
  union {
    ValueComponents components;
    int64_t i;
    void *p;
  };
};

#define RT_PREFIX "_ocamlrt."
#define DECLARE_RT(NAME, RETURN_TYPE, ...) \
  RETURN_TYPE NAME(__VA_ARGS__) __asm__(RT_PREFIX #NAME);
#define DECLARE_RT_AS(NAME, ASM_NAME, RETURN_TYPE, ...) \
  RETURN_TYPE NAME(__VA_ARGS__) __asm__(RT_PREFIX ASM_NAME);

DECLARE_RT(unit, void, Value);
DECLARE_RT(print_float, Value, Value p);
DECLARE_RT(print_int, Value, Value p);
DECLARE_RT(box_convert_i64_f64, Value, Value p);
DECLARE_RT(box_convert_f64_i64, Value, Value p);
DECLARE_RT(embox_i64, Value, int64_t i64);
DECLARE_RT(embox_f64, Value, double f64);
DECLARE_RT(embox_string, Value, const char *str);
DECLARE_RT_AS(ocaml_printf, "Printf.printf", Value, const char *format, ...);
}
