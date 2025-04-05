#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Parse/AST.h"
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include "ocamlc2/Support/Utils.h"
#define DEBUG_TYPE "mlirgen"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
using InsertionGuard = mlir::ocaml::OcamlOpBuilder::InsertionGuard;

static std::string pathToString(llvm::ArrayRef<std::string> path) {
  TRACE();
  assert(path.size() > 0);
  static std::set<std::string> savedStrings;
  std::string result = path.front();
  for (auto &part : llvm::make_range(path.begin() + 1, path.end())) {
    result += "." + part;
  }
  assert(result != "");
  DBGS(result << "\n");
  auto [iterator, _] = savedStrings.insert(std::move(result));
  return *iterator;
}

static mlir::FailureOr<std::string> patternToIdentifier(std::vector<std::string> path) {
  TRACE();
  if (path.size() == 0) {
    return mlir::failure();
  }
  return pathToString(path);
}

static mlir::FailureOr<std::string> patternToIdentifier(ASTNode const& pattern) {
  TRACE();
  if (auto *valuePattern = llvm::dyn_cast<ValuePatternAST>(&pattern)) {
    return valuePattern->getName();
  }
  return mlir::failure();
}

void MLIRGen2::initializeTypeConstructors() {
  typeConstructors.insert("int", {
    [](MLIRGen2 &gen, llvm::ArrayRef<mlir::Type> params) {
      return gen.builder.emboxType(gen.builder.getI64Type());
    }
  });
  typeConstructors.insert("bool", {
    [](MLIRGen2 &gen, llvm::ArrayRef<mlir::Type> params) {
      return gen.builder.emboxType(gen.builder.getI1Type());
    }
  });
  typeConstructors.insert("float", {
    [](MLIRGen2 &gen, llvm::ArrayRef<mlir::Type> params) {
      return gen.builder.emboxType(gen.builder.getF64Type());
    }
  });
  typeConstructors.insert("unit", {
    [](MLIRGen2 &gen, llvm::ArrayRef<mlir::Type> params) {
      return gen.builder.getUnitType();
    }
  });
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(NumberExprAST const& node) {
  TRACE();
  StringRef textRef = node.getValue();
  int result;
  textRef.getAsInteger(10, result);
  auto op = builder.create<mlir::arith::ConstantIntOp>(loc(&node), result, 64);
  auto box = builder.createEmbox(loc(&node), op.getResult())->getResult(0);
  return box;
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ValuePathAST const& node) {
  TRACE();
  auto name = pathToString(node.getPath());
  auto value = getVariable(name, loc(&node));
  if (failed(value)) {
    return mlir::emitError(loc(&node))
        << "Variable '" << name << "' not found";
  }
  return value;
}

mlir::FailureOr<std::string> MLIRGen2::getApplicatorName(ASTNode const& node) {
  TRACE();
  if (auto *path = llvm::dyn_cast<ValuePathAST>(&node)) {
    return pathToString(path->getPath());
  }
  if (auto *path = llvm::dyn_cast<ConstructorPathAST>(&node)) {
    return pathToString(path->getPath());
  }
  return mlir::emitError(loc(&node))
      << "Unknown AST node type: " << ASTNode::getName(node);
}

mlir::FailureOr<mlir::Value> MLIRGen2::genRuntime(llvm::StringRef name, ApplicationExprAST const& node) {
  TRACE();
  SmallVector<mlir::Value> convertedArgs;
  if (name == "print_int") {
    auto arg = gen(*node.getArgument(0));
    if (mlir::failed(arg)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate argument for " << name;
    }
    auto type = builder.emboxType(builder.getI64Type());
    convertedArgs.push_back(
        builder.createConvert(loc(&node), *arg, type)->getResult(0));
    auto call = builder.createCallIntrinsic(loc(&node), "print_int", convertedArgs);
    return call->getResult(0);
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> MLIRGen2::declareVariable(llvm::StringRef name, mlir::Value value, mlir::Location loc) {
  static std::set<std::string> savedNames;
  auto [iterator, _] = savedNames.insert(std::string(name));
  DBGS("declaring '" << name << "' of type " << value.getType() << "\n");
  if (variables.count(*iterator)) {
    return mlir::emitError(loc)
        << "Variable '" << name << "' already declared";
  }
  variables.insert(*iterator, value);
  return value;
}

mlir::FailureOr<mlir::Value> MLIRGen2::getVariable(llvm::StringRef name, mlir::Location loc) {
  DBGS(name << "\n");
  if (not variables.count(name)) {
    return mlir::emitError(loc)
        << "Variable '" << name << "' not declared";
  }
  return variables.lookup(name);
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ParenthesizedExpressionAST const& node) {
  TRACE();
  return gen(*node.getExpression());
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(InfixExpressionAST const& node) {
  TRACE();
  auto rhs = gen(*node.getRHS());
  if (mlir::failed(rhs)) {
    return mlir::emitError(loc(&node))
        << "Failed to generate right hand side of infix expression '" << node.getOperator() << "'";
  }
  auto lhs = gen(*node.getLHS());
  if (mlir::failed(lhs)) {
    return mlir::emitError(loc(&node))
        << "Failed to generate left hand side of infix expression '" << node.getOperator() << "'";
  }
  // auto op = builder.createInfix(loc(&node), *lhs, node.getOperator(), *rhs);
  return lhs;
}

mlir::FailureOr<TypeConstructor> MLIRGen2::getTypeConstructor(ASTNode const& node) {
  TRACE();
  if (auto *typeConstructorPath = llvm::dyn_cast<TypeConstructorPathAST>(&node)) {
    auto maybeName = patternToIdentifier(typeConstructorPath->getPath());
    if (failed(maybeName)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate identifier for type constructor: " << pathToString(typeConstructorPath->getPath());
    }
    auto name = *maybeName;
    DBGS("type constructor '" << name << "'\n");
    if (typeConstructors.count(name)) {
      return typeConstructors.lookup(name);
    }
    return mlir::emitError(loc(&node))
        << "Type constructor " << name << " not found";
  }
  return mlir::failure();
}

mlir::FailureOr<std::pair<std::vector<std::string>, std::vector<mlir::Type>>>
MLIRGen2::processParameters(std::vector<std::unique_ptr<ASTNode>> const &parameters) {
  TRACE();
  std::vector<std::string> parameterNames;
  std::vector<mlir::Type> parameterTypes;
  for (auto &parameter : parameters) {
    if (auto *typedPattern = llvm::dyn_cast<TypedPatternAST>(parameter.get())) {
      auto maybeName = patternToIdentifier(*typedPattern->getPattern());
      if (failed(maybeName)) {
        return mlir::emitError(loc(parameter.get()))
            << "Failed to generate identifier for parameter: " << ASTNode::getName(*parameter);
      }
      parameterNames.push_back(*maybeName);
      auto ctor = getTypeConstructor(*typedPattern->getType());
      if (failed(ctor)) {
        return mlir::emitError(loc(parameter.get()))
            << "Failed to generate type constructor for parameter: " << pathToString(typedPattern->getType()->getPath());
      }
      parameterTypes.push_back(ctor->constructor(*this, {}));
      DBGS("parameter '" << *maybeName << "' of type " << parameterTypes.back() << "\n");
    } else {
      return mlir::emitError(loc(parameter.get()))
          << "Unknown parameter type: " << ASTNode::getName(*parameter);
    }
  }
  return std::make_pair(parameterNames, parameterTypes);
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(LetBindingAST const& node) {
  TRACE();
  auto location = loc(&node);
  auto name = node.getName();
  DBGS("let binding '" << name << "'\n");
  auto &parameters = node.getParameters();
  if (parameters.empty()) {
    // No parameters, just generate the body and assign to the name
    auto body = gen(*node.getBody());
    if (mlir::failed(body)) {
      return mlir::failure();
    }
    return declareVariable(name, *body, location);
  }
  else {
    // Function definition
    mlir::Type returnType;
    if (node.getReturnType()) {
      auto ctor = getTypeConstructor(*node.getReturnType());
      if (failed(ctor)) {
        return mlir::emitError(loc(&node))
            << "Failed to generate return type: " << pathToString(node.getReturnType()->getPath());
      }
      returnType = ctor->constructor(*this, {});
    } else {
      returnType = builder.getUnitType();
    }
    auto maybeProcessedParameters = processParameters(parameters);
    if (mlir::failed(maybeProcessedParameters)) {
      return mlir::emitError(loc(&node))
          << "Failed to determine types for parameters for function: " << name;
    }
    auto [parameterNames, parameterTypes] = *maybeProcessedParameters;
    auto functionType = builder.getFunctionType(parameterTypes, returnType);
    {
      VariableScope scope(variables);
      InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module->getBody());
      auto function = builder.create<mlir::func::FuncOp>(
          location, name, functionType);
      function.setPrivate();
      auto *entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      assert(entryBlock->getArguments().size() == parameterNames.size());
      assert(entryBlock->getArguments().size() == parameterTypes.size());
      for (auto [arg, name, type] : llvm::zip(entryBlock->getArguments(), parameterNames, parameterTypes)) {
        DBGS("\n" << name << " : " << type << "\n");
        if (failed(declareVariable(name, arg, location))) {
          return mlir::emitError(location)
              << "Failed to declare variable: " << name;
        }
      }
      auto body = gen(*node.getBody());
      if (mlir::failed(body)) {
        return mlir::failure();
      }
      builder.create<mlir::func::ReturnOp>(location, *body);
    }
    DBGS("generated function\n");
    return mlir::Value{};
  }
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ValueDefinitionAST const& node) {
  TRACE();
  mlir::FailureOr<mlir::Value> last;
  for (auto &binding : node.getBindings()) {
    if (last = gen(*binding); mlir::failed(last)) {
      return mlir::failure();
    }
  }
  return last;
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ApplicationExprAST const& node) {
  TRACE();
  auto maybeName = getApplicatorName(*node.getFunction());
  if (mlir::failed(maybeName)) {
    return mlir::emitError(loc(&node))
        << "Unknown AST node type: " << ASTNode::getName(node);
  }
  auto name = *maybeName;
  auto function = module->lookupSymbol<mlir::func::FuncOp>(name);
  if (function) {
    llvm::SmallVector<mlir::Value> args;
    for (size_t i = 0; i < node.getNumArguments(); ++i) {
      if (auto arg = gen(*node.getArgument(i)); succeeded(arg)) {
        args.push_back(*arg);
      } else {
        return mlir::emitError(loc(&node))
            << "Failed to generate argument " << i << " for applicator " << name;
      }
    }
    return builder.createCall(loc(&node), function, args)->getResult(0);
  }

  if (auto runtimeCall = genRuntime(name, node); succeeded(runtimeCall)) {
    return runtimeCall;
  }

  return mlir::emitError(loc(&node))
      << "Applicator " << name << " not found and is not a know builtin";
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ForExpressionAST const& node) {
  TRACE();
  auto location = loc(&node);
  auto loopVar = node.getLoopVar();
  auto startExpr = gen(*node.getStartExpr());
  auto endExpr = gen(*node.getEndExpr());

  if (mlir::failed(startExpr) || mlir::failed(endExpr)) {
    return mlir::emitError(location)
        << "Failed to generate start or end expression for for loop";
  }
  auto step = builder
                  .create<mlir::arith::ConstantIntOp>(
                      location, node.getIsDownto() ? -1 : 1, 64)
                  .getResult();
  auto startOp =
      builder.createConvert(location, *startExpr, builder.getI64Type())
          ->getResult(0);
  auto endOp = builder.createConvert(location, *endExpr, builder.getI64Type())
                   ->getResult(0);
  auto forOp = builder.create<mlir::scf::ForOp>(location, startOp, endOp, step);
  {
    InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    auto iterVar = forOp.getInductionVar();
    if (failed(declareVariable(loopVar, iterVar, location))) {
      return mlir::emitError(location)
          << "Failed to declare loop variable: " << loopVar;
    }

    auto body = gen(*node.getBody());
    if (mlir::failed(body)) {
      return mlir::failure();
    }
  }
  return mlir::Value{};
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(LetExpressionAST const& node) {
  TRACE();
  VariableScope scope(variables);
  auto location = loc(&node);
  assert(node.getBinding());
  assert(node.getBody());
  auto binding = gen(*node.getBinding());
  if (mlir::failed(binding)) {
    return mlir::emitError(location)
        << "Failed to generate binding for let expression";
  }
  auto body = gen(*node.getBody());
  if (mlir::failed(body)) {
    return mlir::emitError(location)
        << "Failed to generate body for let expression";
  }
  return *body;
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ASTNode const& node) {
  TRACE();
  if (auto *exprItem = llvm::dyn_cast<ExpressionItemAST>(&node)) {
    return gen(*exprItem);
  } else if (auto *valueDef = llvm::dyn_cast<ValueDefinitionAST>(&node)) {
    return gen(*valueDef);
  } else if (auto *application = llvm::dyn_cast<ApplicationExprAST>(&node)) {
    return gen(*application);
  } else if (auto *number = llvm::dyn_cast<NumberExprAST>(&node)) {
    return gen(*number);
  } else if (auto *valuePath = llvm::dyn_cast<ValuePathAST>(&node)) {
    return gen(*valuePath);
  } else if (auto *forExpr = llvm::dyn_cast<ForExpressionAST>(&node)) {
    return gen(*forExpr);
  } else if (auto *letExpr = llvm::dyn_cast<LetExpressionAST>(&node)) {
    return gen(*letExpr);
  } else if (auto *parenthesizedExpr = llvm::dyn_cast<ParenthesizedExpressionAST>(&node)) {
    return gen(*parenthesizedExpr);
  } else if (auto *infixExpr = llvm::dyn_cast<InfixExpressionAST>(&node)) {
    return gen(*infixExpr);
  }
  return mlir::emitError(loc(&node))
      << "Unknown AST node type: " << ASTNode::getName(node);
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ExpressionItemAST const& node) {
  TRACE();
  return gen(*node.getExpression());
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(CompilationUnitAST const& node) {
  TRACE();
  VariableScope scope(variables);
  TypeConstructorScope typeConstructorScope(typeConstructors);
  initializeTypeConstructors();
  mlir::Location l = loc(&node);
  mlir::FunctionType mainFuncType =
      builder.getFunctionType({}, {builder.getI32Type()});

  auto mainFunc = builder.create<mlir::func::FuncOp>(
      l, "main", mainFuncType);
  auto *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  mlir::FailureOr<mlir::Value> res;
  for (auto &decl : node.getItems()) {
    TRACE();
    res = gen(*decl);
    if (mlir::failed(res)) {
      return mlir::failure();
    }
  }
  mlir::Value returnValue = builder.create<mlir::arith::ConstantIntOp>(l, 0, 32);
  builder.create<mlir::func::ReturnOp>(l, returnValue);
  return returnValue;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen2::gen() {
  TRACE();
  module = builder.create<mlir::ModuleOp>(loc(compilationUnit.get()), "ocamlc2");
  auto *ast = compilationUnit.get();
  auto compilationUnit = llvm::dyn_cast<CompilationUnitAST>(ast);
  if (!compilationUnit) {
    return mlir::emitError(loc(ast))
        << "Compilation unit not found";
  }
  builder.setInsertionPointToStart(module->getBody());
  auto res = gen(*compilationUnit);
  if (mlir::failed(res)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    return mlir::failure();
  }

  if (mlir::failed(module->verify())) {
    DBGS("Failed to verify MLIR for compilation unit\n");
    return mlir::failure();
  }

  return std::move(module);
}
