#include "ocamlc2/Support/Colors.h"
#ifdef DEBUG_TYPE
// #define FUNCTION_NAME __PRETTY_FUNCTION__
#define FUNCTION_NAME __func__
#define DBG(X)                                                                 \
  if (ocamlc2::debug_enabled(DEBUG_TYPE)) {                                    \
    X;                                                                         \
  }
#define DBGS(...)                                                              \
  [&](const auto *__ajm_func, auto __ajm_line) {                               \
    if (ocamlc2::debug_enabled(DEBUG_TYPE)) {                                  \
      llvm::errs() << "[" << ocamlc2::ANSIColors::blue() << DEBUG_TYPE << ":"  \
                   << __ajm_line << ":" << __ajm_func                          \
                   << ocamlc2::ANSIColors::reset() << "]" << " "               \
                   << ocamlc2::ANSIColors::italic() << __VA_ARGS__             \
                   << ocamlc2::ANSIColors::reset();                            \
    }                                                                          \
  }(FUNCTION_NAME, __LINE__)
#else
#error "DEBUG_TYPE is not defined"
#endif
#define TRACE() DBGS("\n")

#ifndef OCAMLC2_DEBUG_H
#define OCAMLC2_DEBUG_H
namespace ocamlc2 {
bool debug_enabled(const std::string &debug_type);
}
#endif
