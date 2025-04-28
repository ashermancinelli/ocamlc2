#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/TSUtil.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <set>
#include <llvm/Support/Casting.h>
#include <ocamlc2/Parse/ScopedHashTable.h>

namespace ocamlc2 {

// Forward declarations
struct TypeExpr;
struct TypeOperator;
struct RecordOperator;
struct FunctionOperator;
struct VariantOperator;
struct TupleOperator;
struct TypeVariable;
struct UnitOperator;
struct VarargsOperator;

struct TypeExpr {
  // clang-format off
  enum Kind {
    Operator = 0b0000'0000'0001,
    Variable = 0b0000'0000'0010,
    Record   = 0b0000'0000'0101,
    Function = 0b0000'0000'1001,
    Tuple    = 0b0000'0001'0001,
    Varargs  = 0b0000'0010'0001,
    Wildcard = 0b0000'0100'0001,
    Variant  = 0b0000'1000'0001,
  };
  // clang-format on
  TypeExpr(Kind kind) : kind(kind) {}
  virtual ~TypeExpr() = default;
  virtual llvm::StringRef getName() const = 0;
  Kind getKind() const { return kind; }
  bool operator==(const TypeExpr& other) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type);
private:
  Kind kind;
};

struct TypeOperator : public TypeExpr {
  TypeOperator(llvm::StringRef name, llvm::ArrayRef<TypeExpr*> args={}) : TypeExpr(Kind::Operator), args(args), name(name) {}
  TypeOperator(TypeExpr::Kind kind, llvm::StringRef name="<error>", llvm::ArrayRef<TypeExpr*> args={}) : TypeExpr(kind), args(args), name(name) {}
  inline llvm::StringRef getName() const override { return name; }
  inline llvm::ArrayRef<TypeExpr*> getArgs() const { return args; }
  static inline bool classof(const TypeExpr* expr) { return (expr->getKind() & Kind::Operator) != 0; }
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op);
  inline TypeExpr* at(size_t index) const { return args[index]; }
  consteval static llvm::StringRef getConstructorOperatorName() { return "V"; }
  consteval static llvm::StringRef getListOperatorName() { return "list"; }
  consteval static llvm::StringRef getArrayOperatorName() { return "array"; }
  consteval static llvm::StringRef getUnitOperatorName() { return "unit"; }
  consteval static llvm::StringRef getWildcardOperatorName() { return "_"; }
  consteval static llvm::StringRef getVarargsOperatorName() { return "varargs!"; }
  consteval static llvm::StringRef getStringOperatorName() { return "string"; }
  consteval static llvm::StringRef getIntOperatorName() { return "int"; }
  consteval static llvm::StringRef getFloatOperatorName() { return "float"; }
  consteval static llvm::StringRef getBoolOperatorName() { return "bool"; }
  consteval static llvm::StringRef getOptionalOperatorName() { return "option"; }
  inline TypeExpr *back() const { return args.back(); }

protected:
  llvm::SmallVector<TypeExpr*> args;
private:
  std::string name;
};

struct WildcardOperator : public TypeOperator {
  WildcardOperator() : TypeOperator(Kind::Wildcard, TypeOperator::getWildcardOperatorName()) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Wildcard; }
};

struct VariantOperator : public TypeOperator {
  using ConstructorType = std::variant<std::pair<llvm::StringRef, FunctionOperator *>, llvm::StringRef>;
  VariantOperator(llvm::StringRef variantName, llvm::ArrayRef<TypeExpr*> args={})
      : TypeOperator(Kind::Variant, variantName, args) {
  }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variant; }
  inline llvm::ArrayRef<ConstructorType> getConstructors() const { return constructors; }
  inline void addConstructor(llvm::StringRef constructorName,
                             FunctionOperator *constructor) {
    constructors.emplace_back(std::make_pair(constructorName, constructor));
  }
  inline void addConstructor(llvm::StringRef constructorName) {
    constructors.emplace_back(constructorName);
  }
  inline std::string decl() const;
private:
  inline llvm::raw_ostream& showCtor(llvm::raw_ostream& os, const ConstructorType& ctor) const;
  llvm::SmallVector<ConstructorType> constructors;
};

struct RecordOperator : public TypeOperator {
  RecordOperator(llvm::StringRef recordName, llvm::ArrayRef<TypeExpr *> args,
                 llvm::ArrayRef<llvm::StringRef> fieldNames)
      : TypeOperator(Kind::Record, recordName, args) {
    std::copy(fieldNames.begin(), fieldNames.end(),
              std::back_inserter(this->fieldNames));
    normalize();
  }
  static inline llvm::StringRef getAnonRecordName() { return "<anon>"; }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Record; }
  inline llvm::ArrayRef<llvm::StringRef> getFieldNames() const { return fieldNames; }
  inline bool isAnonymous() const { return getName() == getAnonRecordName(); }
  inline bool isOpen() const {
    return llvm::any_of(
        getArgs(), [](auto *arg) { return llvm::isa<WildcardOperator>(arg); });
  }
  inline std::string decl() const {
    std::string s;
    llvm::raw_string_ostream ss(s);
    auto zipped = llvm::zip(fieldNames, args);
    auto first = zipped.begin();
    auto [name, type] = *first;
    ss << "{" << name << ":" << *type;
    for (auto [name, type] : llvm::drop_begin(zipped)) {
      ss << "; " << name << ":" << *type;
    }
    ss << '}';
    return ss.str();
  }
private:
  llvm::SmallVector<llvm::StringRef> fieldNames;
  void normalize() {
    auto zipped = llvm::to_vector(llvm::zip(args, fieldNames));
    llvm::sort(zipped, [](auto a, auto b) {
      return std::get<1>(a) < std::get<1>(b);
    });
    args = llvm::map_to_vector(
        zipped, [](auto a) -> TypeExpr * { return std::get<0>(a); });
    fieldNames = llvm::map_to_vector(
        zipped, [](auto a) -> llvm::StringRef { return std::get<1>(a); });
  }
};

struct VarargsOperator : public TypeOperator {
  VarargsOperator() : TypeOperator(Kind::Varargs, TypeOperator::getVarargsOperatorName()) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Varargs; }
};

struct FunctionOperator : public TypeOperator {
  FunctionOperator(
      llvm::ArrayRef<TypeExpr *> args,
      llvm::ArrayRef<ParameterDescriptor> descs = {})
      : TypeOperator(Kind::Function, "function", args) {
    if (descs.empty()) {
      this->parameterDescriptors = llvm::SmallVector<ParameterDescriptor>(args.size() - 1);
    } else {
      assert(descs.size() == args.size() - 1 && "parameter descriptors and arguments must be the same size");
      std::copy(descs.begin(), descs.end(), std::back_inserter(parameterDescriptors));
    }
  }
  llvm::SmallVector<ParameterDescriptor> parameterDescriptors;
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Function; }
  inline bool isVarargs() const {
    for (auto *arg : getArgs()) {
      if (llvm::isa<VarargsOperator>(arg)) {
        return true;
      }
    }
    return false;
  }
};

struct TupleOperator : public TypeOperator {
  TupleOperator(llvm::ArrayRef<TypeExpr*> args) : TypeOperator(Kind::Tuple, "*", args) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Tuple; }
};

struct UnitOperator : public TypeOperator {
  UnitOperator() : TypeOperator(TypeOperator::getUnitOperatorName()) {}
  static inline bool classof(const TypeExpr *expr) {
    if (expr->getKind() == Kind::Operator) {
      auto *op = llvm::cast<TypeOperator>(expr);
      return op->getName() == TypeOperator::getUnitOperatorName();
    }
    return false;
  }
};

struct TypeVariable : public TypeExpr {
  TypeVariable();
  inline llvm::StringRef getName() const override { 
    if (not name) {
      name = std::string("'t" + std::to_string(id));
    }
    return *name;
  }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variable; }
  inline bool instantiated() const { return instance != nullptr; }
  bool operator==(const TypeVariable& other) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var);
  TypeExpr* instance = nullptr;
private:
  int id;
  mutable std::optional<std::string> name = std::nullopt;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op) {
  auto args = op.getArgs();
  auto name = op.getName().str();
  if (args.empty()) {
    return os << name;
  }
  if (args.size() == 1) {
    return os << *args.front() << ' ' << name;
  }
  os << '(' << *args.front();
  for (auto *arg : args.drop_front()) {
    os << ", " << *arg;
  }
  return os << ") " << name;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TupleOperator& tuple) {
  auto args = tuple.getArgs();
  os << *args.front();
  for (auto *arg : args.drop_front()) {
    os << " * " << *arg;
  }
  return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const RecordOperator& record) {
  auto fieldNames = record.getFieldNames();
  auto fieldTypes = record.getArgs();
  assert(fieldNames.size() == fieldTypes.size() && "field names and field types must be the same size");
  if (record.getName() == RecordOperator::getAnonRecordName()) {
    return os << record.decl();
  }
  return os << record.getName();
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionOperator& func) {
  const auto argList = func.getArgs();
  const auto *returnType = argList.back();
  if (!returnType) return os << "<error>";
  const auto args = argList.drop_back();
  assert(!args.empty() && "function type must have at least one argument");
  const auto &descs = func.parameterDescriptors;
  assert(args.size() == descs.size() && "argument list and parameter descriptors must be the same size");
  auto argIter = llvm::zip(descs, args);
  os << '(';
  auto showArg = [&](auto desc, auto *arg) -> llvm::raw_ostream& {
    if (desc.isOptional()) {
      os << "?";
    }
    if (desc.isNamed()) {
      os << desc.label.value() << ":";
    }
    return os << *arg;
  };
  for (auto [desc, arg] : argIter) {
    if (!arg) return os << "<error>";
    showArg(desc, arg) << " -> ";
  }
  if (!returnType) return os << "<error>";
  os << *returnType;
  return os << ')';
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const VariantOperator& variant) {
  for (auto *arg : variant.getArgs()) {
    os << *arg << " ";
  }
  os << variant.getName();
  return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var) {
  if (var.instantiated()) {
    os << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type) {
  if (auto *fo = llvm::dyn_cast<FunctionOperator>(&type)) {
    os << *fo;
  } else if (auto *vo = llvm::dyn_cast<VariantOperator>(&type)) {
    os << *vo;
  } else if (auto *to = llvm::dyn_cast<TupleOperator>(&type)) {
    os << *to;
  } else if (auto *ro = llvm::dyn_cast<RecordOperator>(&type)) {
    os << *ro;
  } else if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}

inline llvm::raw_ostream& VariantOperator::showCtor(llvm::raw_ostream& os, const ConstructorType& ctor) const {
  if (std::holds_alternative<llvm::StringRef>(ctor)) {
    os << std::get<llvm::StringRef>(ctor);
  } else {
    auto [name, fo] = std::get<std::pair<llvm::StringRef, FunctionOperator *>>(ctor);
    os << name << " of ";
    if (fo->getArgs().size() == 2) {
      os << *fo->getArgs().front();
    } else {
      os << "(" << *fo->getArgs().front();
      for (auto *arg : fo->getArgs().drop_front().drop_back()) {
        os << " * " << *arg;
      }
      os << ")";
    }
  }
  return os;
}

inline std::string VariantOperator::decl() const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  auto first = constructors.front();
  showCtor(ss, first);
  for (auto ctor : llvm::drop_begin(constructors)) {
    ss << " | ";
    showCtor(ss, ctor);
  }
  return s;
}

}
