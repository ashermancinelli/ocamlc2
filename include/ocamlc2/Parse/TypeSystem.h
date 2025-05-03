#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/Environment.h"
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
struct ModuleOperator;
struct SignatureOperator;
struct FunctorOperator;

struct TypeExpr {
  // clang-format off
  enum Kind {
    Operator  = 0b0000'0000'0001,
    Variable  = 0b0000'0000'0010,
    Record    = 0b0000'0000'0101,
    Function  = 0b0000'0000'1001,
    Tuple     = 0b0000'0001'0001,
    Varargs   = 0b0000'0010'0001,
    Wildcard  = 0b0000'0100'0001,
    Variant   = 0b0000'1000'0001,
    Signature = 0b0001'0000'0001,
    Module    = 0b0010'0000'0001,
    Functor   = 0b0100'0000'0001,
  };
  // clang-format on
  TypeExpr(Kind kind) : kind(kind) {}
  virtual ~TypeExpr() = default;
  virtual llvm::StringRef getName() const = 0;
  Kind getKind() const { return kind; }
  bool operator==(const TypeExpr& other) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type);
protected:
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
  consteval static llvm::StringRef getFunctorOperatorName() { return "functor"; }
  inline TypeExpr *back() const { return args.back(); }

protected:
  llvm::SmallVector<TypeExpr*> args;
private:
  llvm::StringRef name;
};

struct WildcardOperator : public TypeOperator {
  WildcardOperator() : TypeOperator(Kind::Wildcard, TypeOperator::getWildcardOperatorName()) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Wildcard; }
};

struct VariantOperator : public TypeOperator {
  using ConstructorType = std::variant<std::pair<llvm::StringRef, FunctionOperator *>, llvm::StringRef>;
  VariantOperator(llvm::StringRef variantName, llvm::ArrayRef<TypeExpr*> args={}, llvm::ArrayRef<ConstructorType> constructors={})
      : TypeOperator(Kind::Variant, variantName, args),
        constructors(constructors) {
  }
  VariantOperator(const VariantOperator &other)
      : TypeOperator(other),
        constructors(other.constructors) {
  }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variant; }
  inline llvm::ArrayRef<ConstructorType> getConstructors() const { return constructors; }
  inline llvm::SmallVector<llvm::StringRef> getConstructorNames() const {
    return llvm::map_to_vector(constructors, [](const auto &ctor) {
      if (std::holds_alternative<std::pair<llvm::StringRef, FunctionOperator *>>(ctor)) {
        return std::get<std::pair<llvm::StringRef, FunctionOperator *>>(ctor).first;
      }
      return std::get<llvm::StringRef>(ctor);
    });
  }
  inline void addConstructor(llvm::StringRef constructorName,
                             FunctionOperator *constructor) {
    constructors.emplace_back(std::make_pair(constructorName, constructor));
  }
  inline void addConstructor(llvm::StringRef constructorName) {
    constructors.emplace_back(constructorName);
  }
  std::string decl() const;
private:
  llvm::raw_ostream& showCtor(llvm::raw_ostream& os, const ConstructorType& ctor) const;
  llvm::SmallVector<ConstructorType> constructors;
};

struct RecordOperator : public TypeOperator {
  RecordOperator(llvm::StringRef recordName, llvm::ArrayRef<TypeExpr *> args, llvm::ArrayRef<TypeExpr *> fieldTypes,
                 llvm::ArrayRef<llvm::StringRef> fieldNames)
      : TypeOperator(Kind::Record, recordName, args),
        fieldNames(fieldNames),
        fieldTypes(fieldTypes) {
    normalize();
  }
  RecordOperator(const RecordOperator &other)
      : TypeOperator(other),
        fieldNames(other.fieldNames),
        fieldTypes(other.fieldTypes) {
    normalize();
  }
  static inline llvm::StringRef getAnonRecordName() { return "<anon>"; }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Record; }
  inline llvm::SmallVector<llvm::StringRef> &getFieldNames() { return fieldNames; }
  inline llvm::SmallVector<TypeExpr *> &getFieldTypes() { return fieldTypes; }
  inline llvm::ArrayRef<llvm::StringRef> getFieldNames() const { return fieldNames; }
  inline llvm::ArrayRef<TypeExpr *> getFieldTypes() const { return fieldTypes; }
  inline bool isAnonymous() const { return getName() == getAnonRecordName(); }
  inline bool isOpen() const {
    return llvm::any_of(
        getArgs(), [](auto *arg) { return llvm::isa<WildcardOperator>(arg); });
  }
  std::string decl(const bool named=false) const;
private:
  llvm::SmallVector<llvm::StringRef> fieldNames;
  llvm::SmallVector<TypeExpr*> fieldTypes;
  llvm::ArrayRef<TypeExpr*> typeArgs;
  void normalize() {
    auto zipped = llvm::to_vector(llvm::zip(fieldTypes, fieldNames));
    llvm::sort(zipped, [](auto a, auto b) {
      return std::get<1>(a) < std::get<1>(b);
    });
    fieldTypes = llvm::map_to_vector(
        zipped, [](auto a) -> TypeExpr * { return std::get<0>(a); });
    fieldNames = llvm::map_to_vector(
        zipped, [](auto a) -> llvm::StringRef { return std::get<1>(a); });
  }
};

struct VarargsOperator : public TypeOperator {
  VarargsOperator() : TypeOperator(Kind::Varargs, TypeOperator::getVarargsOperatorName()) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Varargs; }
};

struct FunctorOperator : public TypeOperator {
  FunctorOperator(
      llvm::StringRef name, llvm::ArrayRef<TypeExpr *> args,
      llvm::ArrayRef<std::pair<llvm::StringRef, SignatureOperator *>>
          moduleParameters)
      : TypeOperator(Kind::Functor, name, args),
        moduleParameters(moduleParameters) {}
  FunctorOperator(const FunctorOperator &other)
      : TypeOperator(other),
        moduleParameters(other.moduleParameters) {}
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Functor; }
  inline void pushModuleParameter(llvm::StringRef name, SignatureOperator *type) {
    moduleParameters.emplace_back(name, type);
  }
  inline llvm::ArrayRef<std::pair<llvm::StringRef, SignatureOperator*>> getModuleParameters() const {
    return moduleParameters;
  }
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctorOperator& functor);
private:
  llvm::SmallVector<std::pair<llvm::StringRef, SignatureOperator*>> moduleParameters;
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

struct SignatureOperator : public TypeOperator {
  using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr *>;
  using TypeVarEnv = llvm::ScopedHashTable<llvm::StringRef, TypeVariable *>;
  struct Export {
    enum Kind {
      Type, Variable, Exception
    };
    Kind kind;
    llvm::StringRef name;
    TypeExpr *type;
    Export(Kind kind, llvm::StringRef name, TypeExpr *type) : kind(kind), name(name), type(type) {}
  };
  
  SignatureOperator(llvm::StringRef signatureName, llvm::ArrayRef<TypeExpr*> args={})
      : TypeOperator(Kind::Signature, signatureName, args),
        rootTypeScope(typeEnv),
        rootVariableScope(variableEnv) {}
  SignatureOperator(const SignatureOperator &other)
      : TypeOperator(other.getKind(), other.getName(), other.getArgs()),
        rootTypeScope(typeEnv),
        rootVariableScope(variableEnv),
        exports(other.exports) {
    initFromExports();
  }
  SignatureOperator(llvm::StringRef newName, const SignatureOperator &other) :
      TypeOperator(other.getKind(), newName, other.getArgs()),
      rootTypeScope(typeEnv),
      rootVariableScope(variableEnv),
      exports(other.exports) {
    initFromExports();
  }

  static inline bool classof(const TypeExpr *expr) {
    return expr->getKind() == Kind::Module || expr->getKind() == Kind::Signature;
  }
  llvm::raw_ostream &showSignature(llvm::raw_ostream &os) const;
  llvm::raw_ostream &showDeclaration(llvm::raw_ostream &os) const;
  inline Env &getTypeEnv() { return typeEnv; }
  inline Env &getVariableEnv() { return variableEnv; }
  virtual TypeExpr *lookupType(llvm::StringRef name) const;
  virtual TypeExpr *lookupType(llvm::ArrayRef<llvm::StringRef> path) const;
  virtual TypeExpr *lookupVariable(llvm::StringRef name) const;
  virtual TypeExpr *lookupVariable(llvm::ArrayRef<llvm::StringRef> path) const;
  inline llvm::ArrayRef<Export> getExports() const { return exports; }
  inline TypeExpr *exportType(llvm::StringRef name, TypeExpr *type) {
    typeEnv.insert(name, type);
    if (typeEnv.getCurScope() == &rootTypeScope) {
      exports.push_back(Export(Export::Kind::Type, name, type));
    }
    return type;
  }
  inline TypeExpr *exportVariable(llvm::StringRef name, TypeExpr *type) {
    variableEnv.insert(name, type);
    if (variableEnv.getCurScope() == &rootVariableScope) {
      exports.push_back(Export(Export::Kind::Variable, name, type));
    }
    return type;
  }
  inline TypeExpr *localType(llvm::StringRef name, TypeExpr *type) {
    typeEnv.insert(name, type);
    return type;
  }
  inline TypeExpr *localVariable(llvm::StringRef name, TypeExpr *type) {
    variableEnv.insert(name, type);
    return type;
  }
  static void useNewline(char newline) {
    SignatureOperator::newline = newline;
  }
  static char newlineCharacter() { return newline; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SignatureOperator &signature);
  friend struct FunctorOperator;

private:
  void initFromExports() {
    for (const auto &e : getExports()) {
      if (e.kind == Export::Kind::Type) {
        typeEnv.insert(e.name, e.type);
      } else if (e.kind == Export::Kind::Variable) {
        variableEnv.insert(e.name, e.type);
      }
    }
  }
  Env typeEnv;
  EnvScope rootTypeScope;
  Env variableEnv;
  EnvScope rootVariableScope;
  llvm::SmallVector<Export> exports;
protected:
  static char newline;
};

struct ModuleOperator : public SignatureOperator {
  ModuleOperator(llvm::StringRef moduleName,
                 llvm::ArrayRef<TypeExpr *> args = {})
      : SignatureOperator(moduleName, args) {
    this->kind = Kind::Module;
  }
  ModuleOperator(const ModuleOperator &other)
      : SignatureOperator(other), openModules(other.openModules) {
    this->kind = Kind::Module;
  }
  static inline bool classof(const TypeExpr *expr) { return expr->getKind() == Kind::Module; }
  inline void openModule(ModuleOperator *module) { openModules.push_back(module); }
  TypeExpr *lookupType(llvm::StringRef name) const override;
  TypeExpr *lookupType(llvm::ArrayRef<llvm::StringRef> path) const override;
  TypeExpr *lookupVariable(llvm::StringRef name) const override;
  TypeExpr *lookupVariable(llvm::ArrayRef<llvm::StringRef> path) const override;
private:
  llvm::SmallVector<ModuleOperator *> openModules;
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
  static inline bool classof(const TypeExpr *expr) {
    return expr->getKind() == Kind::Variable;
  }
  inline bool instantiated() const { return instance != nullptr; }
  bool operator==(const TypeVariable &other) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const TypeVariable &var);
  TypeExpr *instance = nullptr;

private:
  int id;
  mutable std::optional<std::string> name = std::nullopt;
};

namespace isa {
inline bool uninstantiatedTypeVariable(const TypeExpr *type) {
  return llvm::isa<TypeVariable>(type) && !llvm::cast<TypeVariable>(type)->instantiated();
}
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TypeOperator &op) {
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

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TupleOperator &tuple) {
  auto args = tuple.getArgs();
  os << *args.front();
  for (auto *arg : args.drop_front()) {
    os << " * " << *arg;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const RecordOperator &record) {
  const auto fieldNames = record.getFieldNames();
  const auto fieldTypes = record.getFieldTypes();
  assert(fieldNames.size() == fieldTypes.size() &&
         "field names and field types must be the same size");
  if (record.getName() == RecordOperator::getAnonRecordName()) {
    return os << record.decl();
  }
  return os << (TypeOperator&)record;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FunctionOperator &func);

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const VariantOperator &variant) {
  for (auto *arg : variant.getArgs()) {
    os << *arg << " ";
  }
  os << variant.getName();
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TypeVariable &var) {
  if (var.instantiated()) {
    os << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SignatureOperator &signature);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeExpr &type);

} // namespace ocamlc2
