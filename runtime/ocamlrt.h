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
  void *ocamlrt_##NAME(RT) __asm__(RT_PREFIX #NAME);

void *ocamlrt_alloc(int tag);
void ocamlrt_printf(const char *format, ...);
void *ocamlrt_print_float(void *p) __asm__("_ocamlrt.print_float");
void *ocamlrt_box_convert_i64_f64(void *p) __asm__("_ocamlrt.box_convert_i64_f64");
void *ocamlrt_box_convert_f64_i64(void *p) __asm__("_ocamlrt.box_convert_f64_i64");
void *ocamlrt_embox_i64(int64_t i64) __asm__("_ocamlrt.embox_i64");
}
