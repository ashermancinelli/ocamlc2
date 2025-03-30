#ifdef DEBUG_TYPE
#define DBG(X) X
#define DBGS(...) llvm::errs() << "[" << DEBUG_TYPE << ":" << __func__ << ":" << __LINE__ << "] " << __VA_ARGS__
#else
#error "DEBUG_TYPE is not defined"
#endif
