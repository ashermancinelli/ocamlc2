#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <sstream>

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;

TypeVariable::TypeVariable() : TypeExpr(Kind::Variable) {
  static int id = 0;
  this->id = id++;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var) {
  return os << var.getName();
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op) {
  os << op.getName();
  for (auto& arg : op.getArgs()) {
    os << ' ' << arg;
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type) {
  if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}
