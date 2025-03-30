#pragma once
#include <cstdint>
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
#define DECLARE_RT(NAME, RT, ...) \
  RT NAME(__VA_ARGS__) __asm__(RT_PREFIX #NAME);

DECLARE_RT(unit, void, Value);
DECLARE_RT(print_float, Value, Value p);
DECLARE_RT(box_convert_i64_f64, Value, Value p);
DECLARE_RT(box_convert_f64_i64, Value, Value p);
DECLARE_RT(embox_i64, Value, int64_t i64);
DECLARE_RT(embox_f64, Value, double f64);
}
