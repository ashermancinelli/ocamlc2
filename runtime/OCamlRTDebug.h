#pragma once

#ifdef OCAMLRT_DEBUG
#define DBGS(...)                                                              \
  if (getenv("OCAMLRT_DEBUG")) {                                               \
    fprintf(stderr, "[%s:%d] ", __func__, __LINE__);                           \
    fprintf(stderr, __VA_ARGS__);                                              \
  }
#else
#define DBGS(...)
#endif
