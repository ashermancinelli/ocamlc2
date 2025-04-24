#pragma once
#include <string>
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
namespace fs = std::filesystem;

mlir::FailureOr<std::string> slurpFile(const std::string &path);

struct StringArena {
  llvm::StringSet<> pool;
  llvm::StringRef save(std::string str) {
    auto [it, _] = pool.insert(str);
    return it->first();
  }
  llvm::StringRef save(std::string_view str) {
    return save(std::string(str));
  }
  llvm::StringRef save(const char *str) {
    return save(std::string_view(str));
  }
};

fs::path moduleNameToFilePath(std::string_view name);
std::string filePathToModuleName(fs::path path);
