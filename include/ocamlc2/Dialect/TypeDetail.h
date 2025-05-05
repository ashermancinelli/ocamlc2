#pragma once

#include <mlir/IR/Attributes.h>
#include <tuple>
#include <llvm/ADT/StringRef.h>

#if 0
namespace mlir::ocaml::detail {
struct VariantTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<llvm::StringRef, llvm::ArrayRef<llvm::StringRef>, llvm::ArrayRef<std::optional<mlir::Type>>>;
  VariantTypeStorage(KeyTy key) : key(key) {}
  static VariantTypeStorage *construct(mlir::TypeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<VariantTypeStorage>()) VariantTypeStorage(key);
  }
  KeyTy key;
};
}

#endif
