#ifdef DEBUG_TYPE
#define DBG(X) X
#define DBGS(...) llvm::errs() << __VA_ARGS__
#else
#error "DEBUG_TYPE is not defined"
#endif
