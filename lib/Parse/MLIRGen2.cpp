#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Parse/AST.h"
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
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
  auto box = builder.createEmbox(loc(&node), op);
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

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ConstructorPathAST const& node) {
  TRACE();
  auto name = pathToString(node.getPath());
  auto function = module->lookupSymbol<mlir::func::FuncOp>(name);
  if (not function) {
    return mlir::emitError(loc(&node))
        << "Function '" << name << "' not found";
  }
  auto call = builder.createCall(loc(&node), function, {});
  return call;
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
        builder.createConvert(loc(&node), *arg, type));
    auto call = builder.createCallIntrinsic(loc(&node), "print_int", convertedArgs);
    return call;
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

mlir::LogicalResult MLIRGen2::declareTypeConstructor(llvm::StringRef name, TypeConstructor constructor, mlir::Location loc) {
  static std::set<std::string> savedNames;
  auto [iterator, _] = savedNames.insert(std::string(name));
  if (typeConstructors.count(*iterator)) {
    return mlir::emitError(loc)
        << "Type constructor '" << name << "' already declared";
  }
  typeConstructors.insert(*iterator, constructor);
  return mlir::success();
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
  auto resultType = mlir::ocaml::resolveTypes(lhs->getType(), rhs->getType(), loc(&node));
  if (mlir::failed(resultType)) {
    return mlir::emitError(loc(&node))
        << "Failed to resolve types for infix expression '" << node.getOperator() << "'";
  }
  auto callOp = builder.createCallIntrinsic(loc(&node), node.getOperator(), {*lhs, *rhs}, *resultType);
  return callOp;
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

mlir::LogicalResult MLIRGen2::genVariantConstructors(mlir::ocaml::VariantType variantType, mlir::Location loc) {
  TRACE();
  auto constructorNames = variantType.getConstructors();
  auto constructorTypes = variantType.getTypes();
  for (auto iter : llvm::enumerate(llvm::zip(constructorNames, constructorTypes))) {
    auto [constructorName, constructorType] = iter.value();
    SmallVector<mlir::Type> argumentTypes;
    bool emptyCtor = false;
    if (auto tupleType = llvm::dyn_cast<mlir::ocaml::TupleType>(constructorType)) {
      argumentTypes = SmallVector<mlir::Type>(tupleType.getTypes());
    } else if (constructorType != builder.getUnitType()) {
      argumentTypes.push_back(constructorType);
    } else {
      emptyCtor = true;
    }
    (void)emptyCtor;
    InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module->getBody());
    auto functionType = builder.getFunctionType(argumentTypes, variantType);
    auto function = builder.create<mlir::func::FuncOp>(loc, constructorName, functionType);
    function.setPrivate();
    auto *entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    auto variantIndex = builder.create<mlir::arith::ConstantIntOp>(loc, iter.index(), 64);
    auto variantIndexBox = builder.createEmbox(loc, variantIndex);
    // if (emptyCtor) {
      SmallVector<mlir::Value> args {variantIndexBox};
      auto variant = builder.create<mlir::ocaml::IntrinsicOp>(loc, variantType, "variant_ctor_empty", args);
      builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{variant});
    // } else {
    //   builder.create<mlir::func::ReturnOp>(loc, variantIndexBox);
    // }
  }
  return mlir::success();
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(TypeBindingAST const& node) {
  TRACE();
  std::string name = node.getName();
  auto location = loc(&node);
  SmallVector<mlir::StringAttr> names;
  SmallVector<mlir::Type> types;
  if (auto *variantDecl = llvm::dyn_cast<VariantDeclarationAST>(node.getDefinition())) {
    auto declarations = gen(*variantDecl);
    if (mlir::failed(declarations)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate type constructor for variant declaration: " << name;
    }
    SmallVector<SmallVector<mlir::Type, 1>> argumentTypes;
    for (auto &ctor : *declarations) {
      auto name = builder.getStringAttr(ctor.first);
      names.push_back(name);
      types.push_back(ctor.second.value_or(builder.getUnitType()));
      if (ctor.second) {
        auto type = ctor.second.value();
        if (auto tupleType = llvm::dyn_cast<mlir::ocaml::TupleType>(type)) {
          argumentTypes.push_back(SmallVector<mlir::Type>(tupleType.getTypes()));
        } else {
          argumentTypes.push_back({type});
        }
      } else {
        argumentTypes.push_back({});
      }
    }

    auto variantType = mlir::ocaml::VariantType::get(
        builder.getContext(), builder.getStringAttr(name), names, types);
    if (failed(genVariantConstructors(variantType, location))) {
      return mlir::emitError(location)
          << "Failed to generate variant constructors for type: " << name;
    }

    auto typeCtor = TypeConstructor{
        [&, name, names, types](
            MLIRGen2 &gen, llvm::ArrayRef<mlir::Type> params) -> mlir::Type {
          return variantType;
        }};
    if (failed(declareTypeConstructor(name, typeCtor, location))) {
      return mlir::emitError(location)
             << "Failed to declare type constructor: " << name;
    }
  } else {
    return mlir::emitError(location)
        << "Unknown type definition: " << ASTNode::getName(node);
  }
  return mlir::Value{};
}


mlir::FailureOr<mlir::Value> MLIRGen2::gen(TypeDefinitionAST const& node) {
  TRACE();
  auto location = loc(&node);
  for (auto &binding : node.getBindings()) {
    if (auto result = gen(*binding); mlir::succeeded(result)) {
      DBGS("generated type constructor\n");
    } else {
      return mlir::emitError(location)
          << "Failed to generate type constructor for type definition";
    }
  }
  return mlir::Value{};
}

mlir::FailureOr<std::optional<mlir::Type>> MLIRGen2::gen(ConstructorDeclarationAST const& node) {
  TRACE();
  if (node.hasSingleType()) {
    auto typeCtor = getTypeConstructor(*node.getOfType());
    if (failed(typeCtor)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate type constructor for constructor declaration: " << node.getName();
    }
    return {typeCtor->constructor(*this, {})};
  } else if (node.getOfTypes().size() > 1) {
    SmallVector<mlir::Type> types;
    for (auto &type : node.getOfTypes()) {
      auto typeCtor = getTypeConstructor(*type);
      if (failed(typeCtor)) {
        return mlir::emitError(loc(&node))
            << "Failed to generate type constructor for constructor declaration: " << node.getName();
      }
      types.push_back(typeCtor->constructor(*this, {}));
    }
    auto type = mlir::ocaml::TupleType::get(builder.getContext(), types);
    return {type};
  }
  return {std::nullopt};
}

mlir::FailureOr<VariantDeclarations> MLIRGen2::gen(VariantDeclarationAST const& node) {
  TRACE();
  auto location = loc(&node);
  auto &ctors = node.getConstructors();
  VariantDeclarations results;
  for (auto &ctor : ctors) {
    auto ctorType = gen(*ctor);
    if (mlir::failed(ctorType)) {
      return mlir::emitError(location)
          << "Failed to generate type constructor for variant declaration: " << ctor->getName();
    }
    results.push_back({ctor->getName(), *ctorType});
  }
  return results;
}

mlir::FailureOr<mlir::Value> MLIRGen2::genPattern(ocamlc2::ASTNode const& node, mlir::Value scrutinee) {
  if (auto *constructorPath = llvm::dyn_cast<ConstructorPathAST>(&node)) {
    return genPattern(*constructorPath, scrutinee);
  } else if (auto *valuePattern = llvm::dyn_cast<ValuePatternAST>(&node)) {
    return genPattern(*valuePattern, scrutinee);
  }
  return mlir::emitError(loc(&node))
      << "Unknown pattern type: " << ASTNode::getName(node);
}

mlir::FailureOr<mlir::Value> MLIRGen2::genPattern(ocamlc2::ConstructorPathAST const& node, mlir::Value scrutinee) {
  auto location = loc(&node);
  auto name = pathToString(node.getPath());
  auto function = module->lookupSymbol<mlir::func::FuncOp>(name);
  if (!function) {
    return mlir::emitError(location)
        << "Unknown constructor: " << name;
  }
  auto args = function.getFunctionType().getResult(0);
  auto variantType = llvm::dyn_cast<mlir::ocaml::VariantType>(args);
  if (!variantType) {
    return mlir::emitError(location)
        << "Constructor is not a variant: " << name;
  }
  if (scrutinee.getType() != variantType) {
    return mlir::emitError(location)
        << "Scrutinee is not a variant: " << name;
  }
  auto variantIndexValue = builder.create<mlir::ocaml::IntrinsicOp>(
      location, builder.emboxType(builder.getI64Type()),
      builder.getStringAttr("variant_get_kind"), mlir::ValueRange{scrutinee});
  auto variantIndex = builder.create<mlir::ocaml::IntrinsicOp>(
      location, builder.getI64Type(), builder.getStringAttr("unbox_i64"),
      mlir::ValueRange{variantIndexValue->getResult(0)});
  auto memberNames = variantType.getConstructors();
  auto iter = llvm::find_if(memberNames, [name](mlir::StringRef memberName) {
    return memberName == name;
  });
  if (iter == memberNames.end()) {
    return mlir::emitError(location)
        << "Unknown constructor: " << name;
  }
  const unsigned index = std::distance(memberNames.begin(), iter);
  auto indexOp = builder.create<mlir::arith::ConstantIntOp>(
      location, index, 64);
  auto comparison = builder.create<mlir::arith::CmpIOp>(
      location, mlir::arith::CmpIPredicate::eq, variantIndex, indexOp);
  return comparison->getResult(0);
}

mlir::FailureOr<mlir::Value> MLIRGen2::genPattern(ocamlc2::ValuePatternAST const& node, mlir::Value scrutinee) {
  auto location = loc(&node);
  auto name = node.getName();
  if (name == "_") {
    return builder.create<mlir::arith::ConstantIntOp>(location, 1, 1)->getResult(0);
  }
  return mlir::emitError(loc(&node))
      << "TODO: handle value pattern for match expression";
}
mlir::FailureOr<mlir::Value> MLIRGen2::genMatchCase(
    MatchCases::const_iterator current, MatchCases::const_iterator end,
    mlir::Value scrutinee, mlir::Type resultType, mlir::Location location) {
  TRACE();
  auto &thisCase = *current;
  auto *pattern = thisCase->getPattern();
  auto matched = genPattern(*pattern, scrutinee);
  if (mlir::failed(matched)) {
    return mlir::emitError(location)
           << "Failed to generate pattern for match case";
  }

  auto *expression = thisCase->getExpression();
  auto ifOp = builder.create<mlir::scf::IfOp>(
      location, /*results=*/mlir::TypeRange{resultType}, *matched, true);
  ifOp->setAttrs(mlir::ocaml::getMatchCaseAttr(builder.getContext()));

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto expressionResult = gen(*expression);
  if (mlir::failed(expressionResult)) {
    return mlir::emitError(location)
           << "Failed to generate expression for match case";
  }

  auto converted =
      builder.createConvert(location, *expressionResult, resultType);
  builder.create<mlir::scf::YieldOp>(location, converted);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  if (++current == end) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(location, 0, 1);
    builder.create<mlir::cf::AssertOp>(
        location, zero, builder.getStringAttr("Match-case is not exhaustive!"));
    auto unitOp = builder.create<mlir::ocaml::UnitOp>(location, builder.getUnitType());
    auto converted = builder.createConvert(location, unitOp, resultType);
    builder.create<mlir::scf::YieldOp>(location, converted);
  } else {
    mlir::Value elseValue;
    {
      InsertionGuard guard(builder);
      auto maybeElseValue =
          genMatchCase(current, end, scrutinee, resultType, location);
      if (mlir::failed(maybeElseValue)) {
        return mlir::emitError(location)
               << "Failed to generate else value for match case";
      }
      elseValue = *maybeElseValue;
    }
    auto converted = builder.createConvert(location, elseValue, resultType);
    builder.create<mlir::scf::YieldOp>(location, converted);
  }

  return ifOp.getResult(0);
}

mlir::FailureOr<mlir::Value> MLIRGen2::genMatchCases(
    MatchCases const& cases,
    mlir::Value scrutinee, mlir::Type resultType, mlir::Location location) {
  TRACE();
  auto matchResult = genMatchCase(cases.begin(), cases.end(), scrutinee, resultType, location);
  if (mlir::failed(matchResult)) {
    return mlir::emitError(location)
        << "Failed to generate match cases";
  }
  DBGS("generated match case\n" << *matchResult << "\n");
  return matchResult;
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(MatchExpressionAST const& node) {
  TRACE();
  VariableScope scope(variables);
  auto location = loc(&node);
  auto maybeScrutinee = gen(*node.getValue());
  if (mlir::failed(maybeScrutinee)) {
    return mlir::emitError(location)
        << "Failed to generate scrutinee for match expression";
  }
  DBGS("generated scrutinee\n" << *maybeScrutinee << '\n');

  auto scrutinee = *maybeScrutinee;
  auto &matchCases = node.getCases();
  auto resultType = builder.getOBoxType(); /* type inference later */

  InsertionGuard guard(builder);
  auto matchCasesResult =
      genMatchCases(matchCases, scrutinee, resultType, location);
  if (mlir::failed(matchCasesResult)) {
    return mlir::emitError(location) << "Failed to generate match cases";
  }
  return *matchCasesResult;
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
    return builder.createCall(loc(&node), function, args);
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
      builder.createConvert(location, *startExpr, builder.getI64Type());
  auto endOp = builder.createConvert(location, *endExpr, builder.getI64Type());
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
  return body;
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ASTNode const& node) {
  TRACE();
  if (auto *exprItem = llvm::dyn_cast<ExpressionItemAST>(&node)) {
    return gen(*exprItem);
  } else if (auto *valueDef = llvm::dyn_cast<ValueDefinitionAST>(&node)) {
    return gen(*valueDef);
  } else if (auto *application = llvm::dyn_cast<ApplicationExprAST>(&node)) {
    return gen(*application);
  } else if (auto *constructorPath = llvm::dyn_cast<ConstructorPathAST>(&node)) {
    return gen(*constructorPath);
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
  } else if (auto *typeDef = llvm::dyn_cast<TypeDefinitionAST>(&node)) {
    return gen(*typeDef);
  } else if (auto *matchExpr = llvm::dyn_cast<MatchExpressionAST>(&node)) {
    return gen(*matchExpr);
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
