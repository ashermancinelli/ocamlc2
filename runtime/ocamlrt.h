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
void *ocamlrt_alloc(int tag);
void ocamlrt_printf(const char *format, ...);

}
