#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <sstream>

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

TypeVariable::TypeVariable() : TypeExpr(Kind::Variable) {
  static int id = 0;
  this->id = id++;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var) {
  if (var.instantiated()) {
    os << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op) {
  os << '(';
  switch (op.getArgs().size()) {
    case 2:
      // Infix notation if we only have two types
      os << op.getArgs()[0] << ' ' << op.getName() << ' ' << op.getArgs()[1];
      break;
    default:
      os << op.getName() << ' ' << '(';
      for (auto *arg : op.getArgs()) {
        os << *arg << ' ';
      }
      break;
  }
  return os << ')';
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type) {
  if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}

bool TypeExpr::operator==(const TypeExpr& other) const {
  if (auto *to = llvm::dyn_cast<TypeOperator>(this)) {
    if (auto *toOther = llvm::dyn_cast<TypeOperator>(&other)) {
      return *to == *toOther;
    }
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(this)) {
    if (auto *tvOther = llvm::dyn_cast<TypeVariable>(&other)) {
      return *tv == *tvOther;
    }
  }
  return false;
}

bool TypeVariable::operator==(const TypeVariable& other) const {
  return id == other.id;
}

TypeExpr* Unifier::infer(const ASTNode* ast) {
  auto *type = inferType(ast);
  ast->typeExpr = type;
  return type;
}

void Unifier::initializeEnvironment() {
  for (auto name : {"int", "float", "bool", "string", "unit"}) {
    env.insert(name, create<TypeOperator>(name));
  }
  auto *T_int = getDeclaredType("int");
  auto *T_unit = getDeclaredType("unit");
  env.insert("print_int", createFunction(T_int, T_unit));
  env.insert("+", createFunction(T_int, createFunction(T_int, T_int)));
}

#if 0
CompilationUnit:
| ExpressionItem:
| | LetExpr:
| | | Binding:
| | | | ValueDefinition:
| | | | | LetBinding: x
| | | | | | Body:
| | | | | | | Number: 0
| | | Body:
| | | | LetExpr:
| | | | | Binding:
| | | | | | ValueDefinition:
| | | | | | | LetBinding: y
| | | | | | | | Body:
| | | | | | | | | Number: 5
| | | | | Body:
| | | | | | ForExpr: i = to
| | | | | | | Start:
| | | | | | | | ValuePath: x
| | | | | | | End:
| | | | | | | | ValuePath: y
| | | | | | | Body:
| | | | | | | | ApplicationExpr:
| | | | | | | | | Function:
| | | | | | | | | | ValuePath: print_int
| | | | | | | | | Arguments:
| | | | | | | | | | ValuePath: i
#endif
static std::string getPath(llvm::ArrayRef<std::string> path) {
  return llvm::join(path, ".");
}

TypeExpr* Unifier::inferType(const ASTNode* ast) {
  DBGS(*ast << '\n');
  if (auto *_ = llvm::dyn_cast<NumberExprAST>(ast)) {
    return getDeclaredType("int");
  } else if (auto *_ = llvm::dyn_cast<StringExprAST>(ast)) {
    return getDeclaredType("string");
  } else if (auto *vp = llvm::dyn_cast<ValuePathAST>(ast)) {
    return getDeclaredType(getPath(vp->getPath()));
  } else if (auto *cu = llvm::dyn_cast<CompilationUnitAST>(ast)) {
    for (auto &item : cu->getItems()) {
      infer(item.get());
    }
  } else if (auto *ei = llvm::dyn_cast<ExpressionItemAST>(ast)) {
    return infer(ei->getExpression());
  } else if (auto *le = llvm::dyn_cast<LetExpressionAST>(ast)) {
    return infer(le->getBody());
  // } else if (auto *fe = llvm::dyn_cast<ForExprAST>(ast)) {
  //   return infer(fe->getBody());
  }
  return nullptr;
}

void Unifier::unify(TypeExpr* a, TypeExpr* b) {
  a = prune(a);
  b = prune(b);
  if (auto *tva = llvm::dyn_cast<TypeVariable>(a)) {
    if (*a != *b) {
      if (isSubType(a, b)) {
        assert(false && "Recursive unification");
      }
      tva->instance = b;
    } else {
      // fallthrough
    }
  } else if (auto *toa = llvm::dyn_cast<TypeOperator>(a)) {
    if (llvm::isa<TypeVariable>(b)) {
      return unify(b, a);
    } else if (auto *tob = llvm::dyn_cast<TypeOperator>(b)) {
      if (toa->getName() != tob->getName() or toa->getArgs().size() != tob->getArgs().size()) {
        llvm::errs() << "Could not unify types: " << *toa << " and " << *tob << '\n';
        assert(false);
      }
      for (auto [aa, bb] : llvm::zip(toa->getArgs(), tob->getArgs())) {
        unify(aa, bb);
      }
    }
  }
}

TypeExpr *Unifier::clone(TypeExpr *type) {
  llvm::DenseMap<TypeExpr *, TypeExpr *> mapping;
  auto *unifier = this;
  auto recurse = [&](this const auto &recurse, TypeExpr *expr) -> TypeExpr * {
    if (auto *op = llvm::dyn_cast<TypeOperator>(expr)) {
      auto args = llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr* arg) {
        return recurse(arg);
      }));
      return unifier->create<TypeOperator>(op->getName(), args);
    } else if (auto *tv = llvm::dyn_cast<TypeVariable>(expr)) {
      if (unifier->isGeneric(tv, concreteTypes)) {
        if (mapping.find(tv) == mapping.end()) {
          mapping[tv] = unifier->create<TypeVariable>();
        }
        return mapping[tv];
      }
    }
    return expr;
  };
  return recurse(type);
}

TypeExpr* Unifier::prune(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type)) {
    if (tv->instantiated()) {
      tv->instance = prune(tv->instance);
      return tv->instance;
    }
  }
  return type;
}

} // namespace ocamlc2

