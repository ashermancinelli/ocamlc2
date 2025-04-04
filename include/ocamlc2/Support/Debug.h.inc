#include "ocamlc2/Support/Colors.h"
#ifdef DEBUG_TYPE
#define DBG(X)                                                                 \
  if (ocamlc2::debug_enabled(DEBUG_TYPE)) {                                    \
    X;                                                                         \
  }
#define DBGS(...)                                                              \
  if (ocamlc2::debug_enabled(DEBUG_TYPE)) {                                    \
    llvm::errs() << "[" << ANSIColors::blue << DEBUG_TYPE << ":" << __func__   \
                 << ":" << __LINE__ << ANSIColors::reset << "]" << " "         \
                 << ANSIColors::italic << __VA_ARGS__ << ANSIColors::reset;    \
  }
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
