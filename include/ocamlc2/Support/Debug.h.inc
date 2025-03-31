#include "ocamlc2/Support/Colors.h"
#ifdef DEBUG_TYPE
#define DBG(X) X
#define DBGS(...)                                                              \
  llvm::errs() << "[" << ANSIColors::blue << DEBUG_TYPE << ":" << __func__     \
               << ":" << __LINE__ << ANSIColors::reset << "]" << " "           \
               << ANSIColors::italic << __VA_ARGS__ << ANSIColors::reset
#else
#error "DEBUG_TYPE is not defined"
#endif
#define TRACE() DBGS("\n")
