#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace ocamlc2 {

inline std::string joinDot(llvm::ArrayRef<llvm::StringRef> path) {
  return llvm::join(path, ".");
}

inline std::string getPath(llvm::ArrayRef<llvm::StringRef> path) {
  return llvm::join(path, ".");
}

inline std::string hashPath(llvm::ArrayRef<llvm::StringRef> path) {
  return llvm::join(path, ".");
}

inline std::string hashPath(std::vector<std::string> path) {
  return llvm::join(path, ".");
}

inline std::string hashPath(llvm::SmallVector<std::string, 8> path) {
  return llvm::join(path, ".");
}

inline std::string hashPath(llvm::SmallVector<llvm::StringRef> path) {
  return llvm::join(path, ".");
}

}
