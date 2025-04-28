#define ERROR(...) DBGS("error\n"), error(__VA_ARGS__, __FILE__, __LINE__)
#define RNULL(...) return ERROR(__VA_ARGS__)
#define RNULL_IF(cond, ...)                                                    \
  if (cond) {                                                                  \
    RNULL(__VA_ARGS__);                                                        \
  }
#define ORNULL(cond)                                                           \
  if (!cond) {                                                                 \
    ERROR("nullptr");                                                         \
    return nullptr;                                                            \
  }
#define ORFAIL(cond)                                                           \
  if (!cond) {                                                                 \
    ERROR("failure");                                                         \
    return failure();                                                          \
  }
#define FAIL(...)                                                              \
  {                                                                            \
    ERROR(__VA_ARGS__);                                                        \
    return failure();                                                          \
  }
#define FAIL_IF(cond, ...)                                                     \
  if (cond) {                                                                  \
    ERROR(__VA_ARGS__);                                                        \
    return failure();                                                          \
  }
#define RNULL_IF_NULL(ptr, ...) RNULL_IF(ptr == nullptr, __VA_ARGS__)
#define SSWRAP(...)                                                            \
  [&] {                                                                        \
    std::string str;                                                           \
    llvm::raw_string_ostream ss(str);                                          \
    ss << __VA_ARGS__;                                                         \
    return ss.str();                                                           \
  }()
#define UNIFY_OR_RNULL(a, b) RNULL_IF(failed(unify(a, b)), "failed to unify")
#define UNIFY_OR_FAIL(a, b) FAIL_IF(failed(unify(a, b)), "failed to unify")
