#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlAttrUtils.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/VisitorHelper.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <utility>
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/MonadHelpers.h"
#define DEBUG_TYPE "MLIRGen3.cpp"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
using InsertionGuard = mlir::ocaml::OcamlOpBuilder::InsertionGuard;

struct ocamlc2::Scope {
  Scope(MLIRGen3 &gen) : gen(gen), variableScope(gen.variables) {}
  MLIRGen3 &gen;
  VariableScope variableScope;

};

using mlir::failed;
using mlir::failure;
using mlir::success;
using mlir::succeeded;
using mlir::LogicalResult;

static StringArena stringArena;

#define SSWRAP(x)                                                              \
  [&] {                                                                        \
    std::string str;                                                           \
    llvm::raw_string_ostream ss(str);                                          \
    ss << x;                                                                   \
    return ss.str();                                                           \
  }()

template <typename T>
mlir::FailureOr<T> mustBe(mlir::FailureOr<std::variant<T, mlir::func::FuncOp>> result) {
  if (failed(result)) {
    return failure();
  }
  if (std::holds_alternative<T>(*result)) {
    return std::get<T>(*result);
  }
  return failure();
}

void MLIRGen3::pushModule(mlir::ocaml::ModuleOp module) {
  DBGS("Pushing module: " << module.getName() << "\n");
  moduleStack.push_back(module);
}

mlir::ocaml::ModuleOp MLIRGen3::popModule() {
  auto module = moduleStack.pop_back_val();
  DBGS("Popping module: " << module.getName() << "\n");
  return module;
}

mlir::ocaml::ModuleOp MLIRGen3::getCurrentModule() const {
  TRACE();
  return moduleStack.back();
}

mlir::ocaml::ModuleOp MLIRGen3::getRootModule() const {
  TRACE();
  return module.get();
}

bool MLIRGen3::shouldAddToModuleType(mlir::Operation *op) {
  // function block arguments should be skipped
  if (op == nullptr) {
    DBGS("op is null\n");
    return false;
  }
  auto parent = op->getParentOp();
  while (parent) {
    if (mlir::isa<mlir::ocaml::ModuleOp, mlir::ocaml::ProgramOp>(parent)) {
      return true;
    }
    if (mlir::isa<mlir::func::FuncOp, mlir::ocaml::BlockOp>(parent)) {
      return false;
    }
    parent = parent->getParentOp();
  }
  return false;
}

mlir::LogicalResult MLIRGen3::shadowGlobalsIfNeeded(llvm::StringRef identifier) {
  return failure();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBindingValueDefinition(const Node patternNode, const Node bodyNode) {
  TRACE();
  auto bodyType = mlirType(patternNode);
  if (failed(bodyType)) {
    return failure();
  }
  UU auto initializerFunctionType = mlir::FunctionType::get(builder.getContext(), {}, {*bodyType});
  // auto identifier = getIdentifierTextFromPattern(patternNode);
  // if (failed(shadowGlobalsIfNeeded(identifier))) {
  //   return failure();
  // }
  auto ip = builder.saveInsertionPoint();
  auto block = builder.create<mlir::ocaml::BlockOp>(loc(bodyNode), *bodyType);
  builder.setInsertionPointToStart(&block.getBody().emplaceBlock());
  return gen(bodyNode) | and_then([&](auto bodyValue) -> mlir::FailureOr<mlir::Value> {
    return mlirType(patternNode) | and_then([&](auto patternType) -> mlir::FailureOr<mlir::Value> {
      if (not ::mlir::ocaml::areTypesCoercible(bodyValue.getType(),
                                               patternType)) {
        return error(patternNode) << "generated type does not agree with "
                                  << "unifier's type for this expression: "
                                  << patternType << " vs " << bodyValue.getType();
      }
      builder.create<mlir::ocaml::YieldOp>(loc(patternNode), bodyValue);
      builder.restoreInsertionPoint(ip);
      if (mlir::isa<mlir::ocaml::UnitType>(patternType)) {
        return bodyValue;
      }
      return declareVariable(patternNode, block.getResult(), loc(patternNode));
    });
  });
}

mlir::FailureOr<mlir::func::FuncOp>
MLIRGen3::genFunctionBody(llvm::StringRef name, mlir::FunctionType funType,
                          mlir::Location location,
                          llvm::ArrayRef<Node> parameters, Node bodyNode) {
  DBGS("Generating function: " << name << "\n");
  InsertionGuard guard(builder);
  VariableScope scope(variables);
  auto env = builder.createEnv(location, getUniqueName((name + "env").str()));
  auto envOp = mlir::cast<mlir::ocaml::EnvOp>(env.getDefiningOp());
  builder.setInsertionPointToStart(getCurrentModule().getBodyBlock());
  auto function =
      builder.create<mlir::func::FuncOp>(location, name, funType);
  function.setPrivate();
  function->setAttr(mlir::ocaml::getEnvironmentAttrName(), envOp.getFor());
  auto *bodyBlock = function.addEntryBlock();
  builder.setInsertionPointToEnd(bodyBlock);
  auto it = function.getArguments().begin();
  auto argRange = llvm::make_range(it, function.getArguments().end());
  for (auto [blockArg, parameter] : llvm::zip(argRange, parameters)) {
    auto parameterNode = parameter.getChildByFieldName("pattern");
    if (failed(declareVariable(parameterNode, blockArg, loc(parameterNode)))) {
      return failure();
    }
  }
  return gen(bodyNode) | and_then([&](auto body) -> mlir::FailureOr<mlir::func::FuncOp> {
    builder.create<mlir::func::ReturnOp>(location, body);
    return {function};
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBinding(const Node node) {
  TRACE();
  const bool isRecursive = isLetBindingRecursive(node.getCursor());
  auto patternNode = node.getChildByFieldName("pattern");
  auto bodyNode = node.getChildByFieldName("body");
  InsertionGuard guard(builder);
  auto *parentScope = variables.getCurScope();
  auto scope = std::make_unique<VariableScope>(variables);
  if (patternNode.getType() == "unit") {
    auto blockOp = builder.create<mlir::ocaml::BlockOp>(loc(node), builder.getUnitType());
    auto &block = blockOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);
    auto bodyValue = gen(bodyNode);
    if (failed(bodyValue)) {
      return failure();
    }
    // Create a fresh unit value for the yield since bodyValue might be from nested scope
    auto unitValue = builder.createUnit(loc(bodyNode));
    builder.create<mlir::ocaml::YieldOp>(loc(bodyNode), unitValue);
    return blockOp.getResult();
  }
  
  auto resultType = mlirType(patternNode);
  if (failed(resultType)) {
    return failure();
  }
  auto identifier = getIdentifierTextFromPattern(patternNode);
  auto letOp = builder.create<mlir::ocaml::LetOp>(loc(node), *resultType, identifier);
  if (isRecursive) {
    letOp->setAttr(mlir::ocaml::getRecursiveAttrName(), builder.getUnitAttr());
  }
  auto &block = letOp.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  auto parameters = getNamedChildren(node, {"parameter"});
  
  // Create a scope for function parameters
  if (!parameters.empty()) {
    DBGS("Adding function arguments for " << identifier << '\n');
    auto closureType = mlir::cast<mlir::ocaml::ClosureType>(*resultType);
    assert(closureType);
    auto functionType = closureType.getFunctionType();
    for (auto [i, parameter] : llvm::enumerate(parameters)) {
      auto paramType = mlirType(parameter);
      if (failed(paramType)) {
        return failure();
      }
      assert(mlir::ocaml::areTypesCoercible(*paramType,
                                            functionType.getInput(i)) &&
             "function type's input does not match the mlir type for the "
             "parameter");
      const auto l = loc(parameter);
      auto ba = block.addArgument(*paramType, l);
      auto res = declareVariable(parameter, ba, l);
      if(failed(res)) {
        return failure();
      }
    }
  }
  
  // For recursive functions, declare self reference inside the function body
  if (isRecursive) {
    auto selfOp =
        builder.create<mlir::ocaml::SelfOp>(loc(patternNode), letOp.getType());
    auto res = declareVariable(patternNode, selfOp, loc(patternNode));
    if (failed(res)) {
      return failure();
    }
  }

  return gen(bodyNode) |
         and_then([&](auto bodyValue) -> mlir::FailureOr<mlir::Value> {
           builder.create<mlir::ocaml::YieldOp>(loc(bodyNode), bodyValue);
           return letOp.getResult();
         }) |
         and_then([&](auto result) -> mlir::FailureOr<mlir::Value> {
           scope.reset();
           return declareVariable(patternNode, result, loc(patternNode),
                                  parentScope);
         });
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirTypeFromBasicTypeOperator(llvm::StringRef name) {
  if (name == "int") {
    return builder.emboxType(builder.getI64Type());
  } else if (name == "float") {
    return builder.emboxType(builder.getF64Type());
  } else if (name == "bool") {
    return builder.emboxType(builder.getI1Type());
  } else if (name == "unit") {
    return builder.getUnitType();
  } else if (name == "string") {
    return builder.getStringType();
  } else if (name == "_") {
    return builder.getOBoxType();
  }
  return failure();
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(const Node node) {
  auto *type = unifierType(node);
  return mlirFunctionType(type, loc(node));
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(ocamlc2::TypeExpr *type, mlir::Location loc) {
  TRACE();
  if (type == nullptr) {
    return error(loc) << "Type for node was not inferred during type inference";
  }
  if (const auto *to = llvm::dyn_cast<ocamlc2::FunctionOperator>(type)) {
    DBGS("Function type: " << *to << '\n');
    const auto typeOperatorArgs = to->getArgs();
    auto args = llvm::drop_end(typeOperatorArgs);
    auto argTypes = llvm::map_to_vector(args, [this, loc](auto *arg) { return mlirType(arg, loc); });
    if (llvm::any_of(argTypes, failed)) {
      return failure();
    }
    SmallVector<mlir::Type> successfulArgTypes;
    auto mappedTypes = llvm::map_to_vector(argTypes, [](auto t) { return *t; });
    llvm::append_range(successfulArgTypes, mappedTypes);
    return mlirType(typeOperatorArgs.back(), loc) |
           and_then([&](auto returnType) -> mlir::FailureOr<mlir::Type> {
             auto functionType = mlir::FunctionType::get(builder.getContext(),
                                                          successfulArgTypes, {returnType});
             return mlir::ocaml::ClosureType::get(functionType);
           });
  }
  return error(loc) << "Could get MLIR type from unified function type: "
                     << SSWRAP(*type);
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirVariantCtorType(ocamlc2::CtorOperator *ctor, mlir::Location loc) {
  TRACE();
  assert(ctor);
  assert(ctor->getArgs().size() == 2 &&
         "Variant ctor can only have 2 arguments, if more it must be a tuple, "
         "if less it's either unit or nullary ctor");
  auto toArgs = ctor->getArgs();
  return mlirType(toArgs.back(), loc) | and_then([&](auto returnType) -> mlir::FailureOr<mlir::Type> {
           auto argType = mlirType(toArgs.front(), loc);
           return argType | and_then([&](auto argType) -> mlir::FailureOr<mlir::Type> {
             auto ft = mlir::FunctionType::get(builder.getContext(), {argType}, {returnType});
             return mlir::ocaml::ClosureType::get(ft);
           });
         });
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(ocamlc2::VariantOperator *type, mlir::Location loc) {
  auto name = type->getName();
  auto constructors = type->getConstructors();
  llvm::SmallVector<mlir::Type> constructorTypes;
  llvm::SmallVector<llvm::StringRef> constructorNames;
  auto visitor = Overload{
    [&](const std::pair<llvm::StringRef, ocamlc2::FunctionOperator *> &constructor) -> LogicalResult {
      auto *constructorTypeOperator = constructor.second;
      auto args = constructorTypeOperator->getArgs();
      if (args.size() != 2) {
        DBGS("constructor: " << *constructorTypeOperator << "\n");
        return error(loc) << "Expected 1 argument for constructor: " << name;
      }
      auto constructorType = mlirType(args.front(), loc);
      if (failed(constructorType)) {
        return failure();
      }
      constructorNames.push_back(constructor.first);
      constructorTypes.emplace_back(std::move(*constructorType));
      return success();
    },
    [&](const llvm::StringRef &constructor) {
      constructorNames.push_back(constructor);
      constructorTypes.emplace_back(builder.getUnitType());
      return success();
    }
  };
  for (auto constructor : constructors) {
    if (failed(std::visit(visitor, constructor))) {
      return failure();
    }
  }
  auto nameAttr = builder.getStringAttr(name);
  auto constructorNameAttrs = builder.createStringAttrVector(constructorNames);
  auto variantType = mlir::ocaml::VariantType::get(
      builder.getContext(), nameAttr, constructorNameAttrs, constructorTypes);
  return variantType;
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirRecordType(ocamlc2::RecordOperator *type, mlir::Location loc) {
  TRACE();
  assert(type);
  auto name = type->getName();
  auto nameAttr = builder.getStringAttr(name);
  auto fieldNames = type->getFieldNames();
  auto fieldNameAttrs = builder.createStringAttrVector(fieldNames);
  auto fieldTypeExprs = type->getFieldTypes();
  auto maybeFieldTypes = llvm::map_to_vector(fieldTypeExprs, [this, loc](auto *fieldTypeExpr) { return mlirType(fieldTypeExpr, loc); });
  if (llvm::any_of(maybeFieldTypes, failed)) {
    return failure();
  }
  llvm::SmallVector<mlir::Type> fieldTypes = llvm::map_to_vector(maybeFieldTypes, [](auto t) { return *t; });
  auto recordType = mlir::ocaml::RecordType::get(builder.getContext(), nameAttr,
                                                 fieldNameAttrs, fieldTypes);
  return recordType;
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(ocamlc2::TypeExpr *type, mlir::Location loc) {
  DBGS("mlirType: " << *type << "\n");
  type = Unifier::pruneEverything(type);
  if (auto *ctor = llvm::dyn_cast<ocamlc2::CtorOperator>(type)) {
    TRACE();
    return mlirVariantCtorType(ctor, loc);
  } else if (auto *funcOperator = llvm::dyn_cast<ocamlc2::FunctionOperator>(type)) {
    TRACE();
    return mlirFunctionType(funcOperator, loc);
  } else if (auto *ro = llvm::dyn_cast<ocamlc2::RecordOperator>(type)) {
    TRACE();
    return mlirRecordType(ro, loc);
  } else if (auto *vo = llvm::dyn_cast<ocamlc2::VariantOperator>(type)) {
    TRACE();
    return mlirType(vo, loc);
  } else if (auto *to = llvm::dyn_cast<ocamlc2::TupleOperator>(type)) {
    TRACE();
    auto args = to->getArgs();
    llvm::SmallVector<mlir::Type> argTypes;
    for (auto *arg : args) {
      auto argType = mlirType(arg, loc);
      if (failed(argType)) {
        return failure();
      }
      argTypes.push_back(*argType);
    }
    return mlir::TupleType::get(builder.getContext(), argTypes);
  } else if (const auto *to = llvm::dyn_cast<ocamlc2::TypeOperator>(type)) {
    TRACE();
    const auto args = to->getArgs();
    if (args.empty()) {
      auto mlirType = mlirTypeFromBasicTypeOperator(to->getName());
      if (failed(mlirType)) {
        error(loc) << "Unknown basic type operator: " << SSWRAP(*type);
        assert(false);
      }
      return mlirType;
    }
    auto name = to->getName();
    DBGS("name: '" << name << "'\n");
    if (name == "array") {
      auto elementType = mlirType(args.front(), loc);
      if (failed(elementType)) {
        return failure();
      }
      return builder.getArrayType(*elementType);
    } else if (name == "list") {
      auto elementType = mlirType(args.front(), loc);
      if (failed(elementType)) {
        return failure();
      }
      return builder.getListType(*elementType);
    } else if (name == "ref") {
      auto elementType = mlirType(args.front(), loc);
      if (failed(elementType)) {
        return failure();
      }
      return mlir::ocaml::ReferenceType::get(*elementType);
    }

    error(loc) << "Unknown type operator: " << SSWRAP(*type);
    assert(false);
  } else if (const auto *tv = llvm::dyn_cast<ocamlc2::TypeVariable>(type)) {
    TRACE();
    if (tv->instantiated()) {
      return mlirType(tv->instance, loc);
    }
    return builder.getOBoxType();
  }
  error(loc) << "Unknown type: " << SSWRAP(*type);
  assert(false);
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(const Node node) {
  auto *type = unifierType(node);
  if (type == nullptr) {
    if (node.getType() == "parameter") {
      return mlirType(node.getChildByFieldName("pattern"));
    }
    return error(node) << "Type for node was not inferred at unification-time";
  }
  if (auto convertedType = typeExprToMlirType.lookup(type)) {
    return convertedType;
  }
  auto convertedType = mlirType(type, loc(node));
  if (failed(convertedType)) {
    return failure();
  }
  typeExprToMlirType[type] = *convertedType;
  return convertedType;
}

mlir::FailureOr<Callee> MLIRGen3::genConstructorPath(const Node node) {
  TRACE();
  return getVariable(node);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genMatchExpression(const Node node) {
  TRACE();
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  auto returnType = *maybeType;
  auto scrutineeNode = node.getChildByFieldName("expression");
  auto maybeScrutineeValue = gen(scrutineeNode);
  if (failed(maybeScrutineeValue)) {
    return failure();
  }
  auto scrutineeValue = *maybeScrutineeValue;
  auto caseNodes = getNamedChildren(node, {"match_case"});
  return [&] -> mlir::FailureOr<mlir::Value> {
    InsertionGuard guard(builder);
    auto regionOp = builder.create<mlir::ocaml::BlockOp>(loc(node), returnType);
    auto &region = regionOp.getRegion();
    auto &returnBlock = region.emplaceBlock();
    returnBlock.addArgument(returnType, loc(node));
    builder.setInsertionPointToEnd(&returnBlock);
    builder.create<mlir::ocaml::YieldOp>(loc(node), returnBlock.getArgument(0));
    auto *matchFailureBlock = builder.createBlock(&returnBlock);
    auto falseOp = builder.create<mlir::arith::ConstantOp>(loc(node), builder.getI1Type(), builder.getBoolAttr(false));
    builder.create<mlir::cf::AssertOp>(loc(node), falseOp, "No match found");
    auto failureReturn = builder.createConvert(loc(node), returnType, falseOp);
    builder.create<mlir::cf::BranchOp>(loc(node), &returnBlock, mlir::ValueRange{failureReturn});
    for (auto caseNode : caseNodes) {
      auto *patternBlock = builder.createBlock(matchFailureBlock);
      auto *bodyBlock = builder.createBlock(matchFailureBlock);
      builder.setInsertionPointToEnd(patternBlock);
      auto patternNode = caseNode.getChildByFieldName("pattern");
      auto bodyNode = caseNode.getChildByFieldName("body");
      auto maybePatternValue = gen(patternNode);
      if (failed(maybePatternValue)) {
        return failure();
      }
      auto patternValue = *maybePatternValue;
      auto matchedPattern = builder.createPatternMatch(
          loc(caseNode), scrutineeValue, patternValue);
      builder.create<mlir::cf::CondBranchOp>(loc(caseNode), matchedPattern, bodyBlock, matchFailureBlock);
      builder.setInsertionPointToEnd(bodyBlock);
      auto bodyValue = gen(bodyNode);
      builder.create<mlir::cf::BranchOp>(loc(bodyNode), &returnBlock,
                                         mlir::ValueRange{*bodyValue});
      matchFailureBlock = patternBlock;
    }
    return {regionOp};
  }();
}

FailureOr<std::tuple<bool, mlir::func::FuncOp>>
MLIRGen3::valueIsFreeInCurrentContext(mlir::Value value) {
  TRACE();
  mlir::func::FuncOp foundValueInFunc;
  mlir::ocaml::ProgramOp foundValueInProgram;
  auto result = [&] -> LogicalResult {
    auto *op = value.getDefiningOp();
    if (op) {
      DBGS("Op\n");
      if (auto f = op->getParentOfType<mlir::func::FuncOp>()) {
        foundValueInFunc = f;
        return success();
      }
      if (auto p = op->getParentOfType<mlir::ocaml::ProgramOp>()) {
        foundValueInProgram = p;
        return success();
      }
      return failure();
    }
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (arg) {
      DBGS("Block argument\n");
      auto parent = arg.getOwner()->getParent()->getParentOfType<mlir::func::FuncOp>();
      if (parent) {
        DBGS("Func: " << parent.getSymName() << "\n");
        foundValueInFunc = parent;
        return success();
      }
      return failure();
    }
    DBGS("No func\n");
    return failure();
  }();
  if (failed(result)) {
    DBGS("No func\n");
    return failure();
  }
  auto *currentBlock = builder.getInsertionBlock();
  auto currentFunc = currentBlock->getParent()->getParentOfType<mlir::func::FuncOp>();

  if (currentFunc == foundValueInFunc) {
    DBGS("Current region\n");
    return {{false, currentFunc}};
  }
  DBGS("Not current region\n");
  return {{true, currentFunc}};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genGlobalForFreeVariable(
    mlir::Value value, llvm::StringRef name, mlir::Location loc) {
  DBGS("gen env capture: " << name << "\n");
  return valueIsFreeInCurrentContext(value) | and_then([&](auto tup) -> mlir::FailureOr<mlir::Value> {
    auto [isFree, currentFunc] = tup;
    if (not isFree) {
      return value;
    }
    return findEnvForFunction(currentFunc) |
           and_then([&](mlir::Value env) -> mlir::LogicalResult {
             InsertionGuard guard(builder);
             builder.setInsertionPointAfterValue(env);
             builder.createEnvCapture(loc, env, name, value);
             return success();
           }) |
           and_then([&]() -> mlir::FailureOr<mlir::Value> {
             InsertionGuard guard(builder);
             builder.setInsertionPointToStart(
                &currentFunc.getFunctionBody().front());
             auto envArg = builder.create<mlir::ocaml::EnvGetCurrentOp>(loc);
             builder.setInsertionPointAfterValue(envArg);
             auto loadCapturedArg =
                builder.createEnvGet(loc, value.getType(), envArg, name);
             return {loadCapturedArg};
           }) |
           and_then([&](mlir::Value loadedValue) -> FailureOr<mlir::Value> {
             DBGS("redeclaring variable: " << name
                                           << " as loaded from environment\n");
             return declareVariable(name, loadedValue, loc);
           });
  });
}

llvm::SmallVector<mlir::ocaml::ModuleOp> MLIRGen3::getModuleSearchPath() const {
  llvm::SmallVector<mlir::ocaml::ModuleOp> path;
  llvm::append_range(path, llvm::reverse(moduleStack));
  // TODO: add open modules to search path
  return path;
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(const Node node) {
  auto path = getTextPathFromValuePath(node);
  DBGS("Getting variable: " << llvm::join(path, ".") << "\n");
  auto found = getVariable(llvm::join(path, "."), loc(node));
  if (succeeded(found)) {
    return found;
  }
  return mlirType(node) | and_then([&](auto type) -> mlir::FailureOr<mlir::Value> {
    auto lookupOp = builder.create<mlir::ocaml::ModuleLookupOp>(loc(node), type, llvm::join(path, "."));
    auto result = lookupOp.getResult();
    return {result};
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(llvm::StringRef name, mlir::Location loc) {
  if (auto value = variables.lookup(name)) {
    return value;
  }
  return failure();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genForExpression(const Node node) {
  TRACE();
  const auto children = getNamedChildren(node);
  const auto inductionVariableNode = children[0];
  const auto startNode = children[1];
  const auto endNode = children[2];
  const auto bodyNode = children[3];
  const auto upOrDownToNode = node.getChild(4);
  const auto stepSize = upOrDownToNode.getType() == "to"       ? 1
                        : upOrDownToNode.getType() == "downto" ? -1
                                                           : 0;
  if (stepSize == 0) {
    return error(node) << "Could not infer step size";
  }
  auto startValue = gen(startNode);
  auto endValue = gen(endNode);
  if (failed(startValue) || failed(endValue)) {
    return error(node) << "Failed to generate bounds for `for` expression";
  }
  mlir::Value start = builder.createUnbox(loc(startNode), *startValue);
  mlir::Value end = builder.createUnbox(loc(endNode), *endValue);
  mlir::Value step = builder.createConstant(loc(upOrDownToNode), builder.getI64Type(),
                                     stepSize);
  auto forOp = builder.create<mlir::scf::ForOp>(loc(node), start, end, step);
  auto bodyBlock = forOp.getBody();
  VariableScope scope(variables);
  builder.setInsertionPointToStart(bodyBlock);
  auto inductionVariable = builder.createEmbox(loc(inductionVariableNode), forOp.getInductionVar());
  auto res = declareVariable(inductionVariableNode, inductionVariable,
                             loc(inductionVariableNode));
  if (failed(res)) {
    return res;
  }
  auto bodyValue = gen(bodyNode);
  if (failed(bodyValue)) {
    return error(node) << "Failed to generate body for `for` expression";
  }
  return builder.createUnit(loc(node));
}

mlir::FailureOr<mlir::Value> MLIRGen3::genCompilationUnit(const Node node) {
  TRACE();
  VariableScope scope(variables);
  builder.setInsertionPointToStart(module->getBodyBlock());
  {
    InsertionGuard guard(builder);
    for (auto child : getNamedChildren(node)) {
      if (child.getType() == "comment") {
        continue;
      }
      if (mlir::failed(gen(child))) {
        return mlir::failure();
      }
    }
  }
  return mlir::Value();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genNumber(const Node node) {
  TRACE();
  auto str = getText(node);
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  long long int intValue;
  double doubleValue;
  if (not str.getAsInteger(10, intValue)) {
    auto constant = builder.createConstant(loc(node), builder.getI64Type(), intValue);
    return builder.createEmbox(loc(node), constant);
  } else if (not str.getAsDouble(doubleValue)) {
    auto constant = builder.createConstant(loc(node), builder.getF64Type(), doubleValue);
    return builder.createEmbox(loc(node), constant);
  }
  return error(node) << "Failed to parse number: " << str;
}

mlir::FailureOr<Callee> MLIRGen3::genBuiltinCallee(const Node node) {
  TRACE();
  auto str = unifier.getTextSaved(node);
  if (str == "ref") {
    return Callee{BuiltinBuilder{[&](mlir::Location loc, mlir::ValueRange args)
                                     -> mlir::FailureOr<mlir::Value> {
      auto ref = builder.create<mlir::ocaml::ReferenceOp>(loc, args[0]);
      return {ref};
    }}};
  } else if (str == "!") {
    return Callee{BuiltinBuilder{[&](mlir::Location loc, mlir::ValueRange args)
                                     -> mlir::FailureOr<mlir::Value> {
      auto ref = builder.create<mlir::ocaml::LoadOp>(loc, args[0]);
      return {ref};
    }}};
  } else if (str == ":=") {
    return Callee{BuiltinBuilder{[&](mlir::Location loc, mlir::ValueRange args)
                                     -> mlir::FailureOr<mlir::Value> {
      auto memref = args[0];
      auto value = args[1];
      builder.create<mlir::ocaml::StoreOp>(loc, value, memref);
      return {mlir::Value()};
    }}};
  }
  return failure();
}

mlir::FailureOr<Callee> MLIRGen3::genCallee(const Node node) {
  auto text = getText(node);
  DBGS("lookup " << text << " in current context\n");
  return getVariable(node) |
         or_else([&]() -> mlir::FailureOr<Callee> {
           DBGS("Looking for builtin callee\n");
           return genBuiltinCallee(node);
         }) |
         or_else([&]() -> mlir::FailureOr<Callee> {
           DBGS("Generating unresolved stub for " << text << "\n");
           return mlirType(node) |
                  and_then([&](mlir::Type type)
                               -> mlir::FailureOr<mlir::FunctionType> {
                    return llvm::TypeSwitch<
                               mlir::Type, mlir::FailureOr<mlir::FunctionType>>(
                               type)
                        .Case<mlir::ocaml::ClosureType>(
                            [](mlir::ocaml::ClosureType closureType)
                                -> mlir::FailureOr<mlir::FunctionType> {
                              return closureType.getFunctionType();
                            })
                        .Case<mlir::FunctionType>(
                            [](mlir::FunctionType functionType)
                                -> mlir::FailureOr<mlir::FunctionType> {
                              return functionType;
                            })
                        .Default([&](mlir::Type type)
                                     -> mlir::FailureOr<mlir::FunctionType> {
                          return error(node)
                                 << "Expected function type for callee: "
                                 << type;
                        });
                  }) |
                  and_then([&](auto functionType) -> mlir::FailureOr<Callee> {
                    InsertionGuard guard(builder);
                    builder.setInsertionPointToStart(getCurrentModule().getBodyBlock());
                    auto callee = builder.create<mlir::func::FuncOp>(
                        loc(node), text, functionType);
                    callee.setPrivate();
                    callee->setAttr(mlir::ocaml::getUnresolvedFunctionAttrName(),
                                    mlir::UnitAttr::get(builder.getContext()));
                    return Callee{callee};
                  });
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetExpression(const Node node) {
  TRACE();
  VariableScope scope(variables);
  InsertionGuard guard(builder);
  auto definitions = getNamedChildren(node, {"value_definition"});
  auto body = node.getChildByFieldName("body");
  auto maybeBodyType = mlirType(body);
  if (failed(maybeBodyType)) {
    return failure();
  }
  auto bodyType = *maybeBodyType;
  auto letExpressionRegion = builder.create<mlir::ocaml::BlockOp>(loc(node), bodyType);
  auto &bodyBlock = letExpressionRegion.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(&bodyBlock);
  for (auto definition : definitions) {
    if (failed(gen(definition))) {
      return failure();
    }
  }
  auto maybeBodyValue = gen(body);
  if (failed(maybeBodyValue)) {
    return failure();
  }

  if (!mlir::isa<mlir::ocaml::UnitType>(bodyType)) {
    auto bodyValue = *maybeBodyValue;
    builder.create<mlir::ocaml::YieldOp>(loc(node), bodyValue);
  } else {
    builder.create<mlir::ocaml::YieldOp>(
        loc(node), builder.createUnit(loc(node)));
  }
  return {letExpressionRegion};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genInfixExpression(const Node node) {
  TRACE();
  const auto children = getNamedChildren(node);
  const auto lhs = children[0];
  const auto rhs = children[2];
  const auto op = children[1];
  auto l = loc(node);
  return genCallee(op) |
         and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
           return gen(lhs) |
                  and_then([&](auto lhsValue) -> mlir::FailureOr<mlir::Value> {
                    return gen(rhs) |
                           and_then([&](auto rhsValue) {
                             return genApplication(l, callee, {lhsValue, rhsValue});
                           });
                  });
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::findEnvForFunctionOrNullEnv(mlir::func::FuncOp funcOp) {
  return findEnvForFunction(funcOp) |
         or_else([&]() -> mlir::FailureOr<mlir::Value> {
           auto env = builder.create<mlir::ocaml::EnvOp>(
               funcOp.getLoc(),
               mlir::ocaml::EnvType::get(builder.getContext()));
           return {env};
         });
}


mlir::FailureOr<std::variant<mlir::ocaml::ProgramOp, mlir::func::FuncOp>>
MLIRGen3::getCurrentFuncOrProgram(mlir::Operation *op) {
  if (op == nullptr) {
    auto *block = builder.getInsertionBlock();
    op = block->getParentOp();
    if (auto program = mlir::dyn_cast<mlir::ocaml::ProgramOp>(op)) {
      return {program};
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      return {func};
    }
    // fallthrough
  }

  if (auto program = op->getParentOfType<mlir::ocaml::ProgramOp>()) {
    return {program};
  } else if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
    return {func};
  }

  return failure();
}

mlir::FailureOr<mlir::Value> MLIRGen3::findEnvForFunction(mlir::func::FuncOp funcOp) {
  TRACE();
  auto envAttr = funcOp->getAttr(mlir::ocaml::getEnvironmentAttrName());
  if (not envAttr) {
    DBGS("no env attr\n");
    return failure();
  }
  FailureOr<mlir::Value> env=failure();
  getModule().walk([&](mlir::ocaml::EnvOp envOp) {
    if (envOp.getFor() == envAttr) {
      env = {envOp};
    }
  });
  return env;
}

mlir::FailureOr<mlir::Value>
MLIRGen3::genCallOrCurry(mlir::Location loc, mlir::Value closure, llvm::ArrayRef<mlir::Value> args) {
  TRACE();
  auto closureType = mlir::cast<mlir::ocaml::ClosureType>(closure.getType());
  auto functionType = closureType.getFunctionType();
  if (functionType.getNumInputs() == args.size()) {
    return {builder.create<mlir::ocaml::CallOp>(loc, closure, args)};
  } else {
    return {builder.create<mlir::ocaml::CurryOp>(loc, closure, args)};
  }
}

mlir::FailureOr<mlir::Value>
MLIRGen3::genApplication(mlir::Location loc, Callee callee,
                         llvm::ArrayRef<mlir::Value> args) {
  TRACE();
  return std::visit(
      Overload{[&](BuiltinBuilder builder) -> mlir::FailureOr<mlir::Value> {
                 return builder(loc, args);
               },
               [&](mlir::func::FuncOp funcOp) -> mlir::FailureOr<mlir::Value> {
                 auto current = getCurrentFuncOrProgram();
                 if (succeeded(current)) {
                   if (std::holds_alternative<mlir::func::FuncOp>(*current)) {
                     auto enclosingFunc =
                         std::get<mlir::func::FuncOp>(*current);
                     if (enclosingFunc == funcOp) {
                       // In this case we are calling ourselves recursively -
                       // don't do anything fancy with the environments!! Just
                       // get the current environment instead of reaching
                       // through the environment to the enclosing scope to get
                       // the environment that way.
                       auto env = builder.create<mlir::ocaml::EnvGetCurrentOp>(loc);
                       auto closure = builder.create<mlir::ocaml::ClosureOp>(loc, funcOp, env);
                       return genApplication(loc, closure, args);
                     }
                   }
                 }
                 return findEnvForFunction(funcOp) |
                        and_then([&](auto env) -> mlir::FailureOr<mlir::Value> {
                          auto closure = builder.create<mlir::ocaml::ClosureOp>(loc, funcOp, env);
                          return genApplication(loc, closure, args);
                        }) | or_else([&]() -> mlir::FailureOr<mlir::Value> {
                          auto closure = builder.create<mlir::ocaml::ClosureOp>(loc, funcOp);
                          return genApplication(loc, closure, args);
                        });
               },
               [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
                 auto closureType = mlir::cast<mlir::ocaml::ClosureType>(value.getType());
                 assert(closureType && "Expected closure type for callee when invoking a value");
                 return genCallOrCurry(loc, value, args);
               }},
      callee);
}

mlir::FailureOr<mlir::Value>
MLIRGen3::genApplicationExpression(const Node node) {
  TRACE();
  auto callee = node.getChildByFieldName("function");
  auto args = SmallVector<Node>{llvm::drop_begin(getNamedChildren(node))};
  DBGS("callee: " << getText(callee) << " args: " << args.size() << '\n');
  return genCallee(callee) |
         and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
           const auto maybeArgs = llvm::to_vector(llvm::map_range(
               args, [this](const Node &arg) { return gen(arg); }));
           if (llvm::any_of(maybeArgs, failed)) {
             return error(node) << "Failed to generate arguments";
           }
           auto argValues = llvm::map_to_vector(
               maybeArgs, [](auto value) { return *value; });
           return genApplication(loc(node), callee, argValues);
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genArrayExpression(const Node node) {
  TRACE();
  auto arrayType = mlirType(node);
  if (failed(arrayType)) {
    return failure();
  }
  auto elementNodes = getNamedChildren(node);
  auto maybeElementTypes = llvm::to_vector(llvm::map_range(
      elementNodes, [this](const Node &arg) { return mlirType(arg); }));
  if (llvm::any_of(maybeElementTypes, failed)) {
    return error(node) << "Failed to generate element types";
  }
  auto elements = llvm::map_to_vector(
      elementNodes, [this](const Node &arg) { return gen(arg); });
  if (llvm::any_of(elements, failed)) {
    return error(node) << "Failed to generate elements";
  }
  auto elementValues =
      llvm::map_to_vector(elements, [](auto value) { return *value; });
  DBGS("array of type " << *arrayType << " with " << elementValues.size()
                        << " elements\n");
  auto array =
      builder.createArrayFromElements(loc(node), *arrayType, elementValues);
  assert(array.getType() == *arrayType);
  return array;
}

llvm::StringRef MLIRGen3::getIdentifierTextFromPattern(const Node node) {
  auto text = getText(node);
  DBGS(text << '\n');
  return text;
}

llvm::StringRef MLIRGen3::getTextStripQuotes(const Node node) {
  TRACE();
  assert(node.getType() == "string");
  return getText(node).drop_front().drop_back();
}

static void pathCollectionHelper(llvm::SmallVector<llvm::StringRef> &textPath, Node node, MLIRGen3 &gen) {
  auto children = getNamedChildren(node);
  if (!children.empty()) {
    for (auto child : children) {
      pathCollectionHelper(textPath, child, gen);
    }
  } else {
    textPath.push_back(gen.getUnifier().getTextSaved(node));
  }
}

llvm::SmallVector<llvm::StringRef> MLIRGen3::getTextPathFromValuePath(Node node) {
  llvm::SmallVector<llvm::StringRef> textPath;
  pathCollectionHelper(textPath, node, *this);
  return textPath;
}

llvm::StringRef MLIRGen3::getTextFromValuePath(Node node) {
  DBGS(node.getSExpr().get() << '\n');
  auto children = getNamedChildren(node);
  auto type = node.getType();
  if (type == "value_path") {
    if (not children.empty() and children[0].getType() == "parenthesized_operator") {
      node = children[0].getNamedChild(0);
    }
  } else if (type == "parenthesized_operator") {
    node = node.getNamedChild(0);
  }
  auto text = getText(node);
  DBGS(text << '\n');
  return text;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genModuleBinding(const Node node) {
  TRACE();
  VariableScope scope(variables);
  assert(node.getType() == "module_binding");
  auto name = node.getNamedChild(0);
  assert(!name.isNull() && "Expected module name");
  assert(name.getType() == "module_name" && "Expected module name");
  auto nameText = getText(name);
  DBGS("Got module name: " << nameText << '\n');
  auto signature = node.getChildByFieldName("module_type");
  DBGS("signature?: " << signature.getSExpr().get() << '\n');
  auto structure = node.getChildByFieldName("body");
  auto moduleParameters = getNamedChildren(node, {"module_parameter"});
  if (not moduleParameters.empty()) {
    return nyi(node) << " functors";
  }
  assert(!structure.isNull());
  InsertionGuard guard(builder);
  pushModule(builder.create<mlir::ocaml::ModuleOp>(loc(node), nameText));
  auto &block = getCurrentModule().getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  return gen(structure) | and_then([&](auto body) -> mlir::FailureOr<mlir::Value> {
    return {popModule()};
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genFieldGetExpression(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  auto record = node.getChildByFieldName("record");
  auto fieldName = node.getChildByFieldName("field");
  auto fieldNameStr = getText(fieldName);
  return gen(record) | and_then([&](auto recordValue) -> mlir::FailureOr<mlir::Value> {
    auto got = builder.create<mlir::ocaml::RecordGetOp>(loc(node), recordValue, fieldNameStr);
    return {got};
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genRecordPattern(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  mlir::Value record = builder.create<mlir::ocaml::UndefOp>(loc(node), *type);
  auto patterns = getNamedChildren(node, {"field_pattern"});
  for (auto pattern : patterns) {
    auto location = loc(pattern);
    auto fieldType = mlirType(pattern);
    if (failed(fieldType)) {
      return failure();
    }
    auto fieldName = pattern.getNamedChild(0);
    auto fieldValueNode = toOptional(pattern.getNamedChild(1));
    auto fieldNameStr = getText(fieldName);
    auto fieldValue = fieldValueNode ? gen(*fieldValueNode) : builder.createPatternVariable(location, *fieldType);
    record = builder.create<mlir::ocaml::RecordSetOp>(location, record, fieldNameStr, *fieldValue);
  }
  return {record};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genRecordExpression(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  mlir::Value record = builder.create<mlir::ocaml::UndefOp>(loc(node), *type);
  auto children = getNamedChildren(node, {"field_expression"});
  llvm::SmallVector<mlir::Value> fields;
  for (auto child : children) {
    auto fieldName = child.getNamedChild(0);
    auto fieldValueNode = child.getNamedChild(1);
    auto fieldValue = gen(fieldValueNode);
    if (failed(fieldValue)) {
      return failure();
    }
    auto fieldNameStr = getText(fieldName);
    record = builder.create<mlir::ocaml::RecordSetOp>(loc(node), record, fieldNameStr, *fieldValue);
  }
  return {record};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genModuleStructure(const Node node) {
  TRACE();
  assert(node.getType() == "structure");
  auto children = getNamedChildren(node);
  for (auto child : children) {
    auto value = gen(child);
    if (failed(value)) {
      return failure();
    }
  }
  return {getCurrentModule()};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genModuleDefinition(const Node node) {
  TRACE();
  assert(node.getType() == "module_definition");
  return genModuleBinding(node.getNamedChild(0));
}

mlir::FailureOr<mlir::Value> MLIRGen3::genExternal(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  auto children = getNamedChildren(node);
  assert(children.size() == 3);
  auto name = children[0];
  auto bindcName = children[2];
  auto nameStr = getTextFromValuePath(name);
  auto bindcNameStr = getTextStripQuotes(bindcName);
  return mlirType(node) |
         and_then([&](auto type) -> mlir::FailureOr<mlir::Value> {
           {
             InsertionGuard guard(builder);
             builder.setInsertionPointToStart(getCurrentModule().getBodyBlock());
             builder.create<mlir::ocaml::ExternalOp>(loc(node), nameStr,
                                                     bindcNameStr, type);
           }
           return mlir::Value{};
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genPrefixExpression(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  auto oper = node.getChildByFieldName("operator");
  auto operand = node.getChildByFieldName("expression");
  return genCallee(oper) | and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
    return gen(operand) | and_then([&](auto operand) -> mlir::FailureOr<mlir::Value> {
      return genApplication(loc(node), callee, {operand});
    });
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genConsExpression(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  assert(llvm::isa<mlir::ocaml::ListType>(*type));
  auto value = node.getChildByFieldName("left");
  auto list = node.getChildByFieldName("right");
  return gen(value) | and_then([&](auto value) -> mlir::FailureOr<mlir::Value> {
    return gen(list) | and_then([&](auto list) -> mlir::FailureOr<mlir::Value> {
      auto consOp = builder.create<mlir::ocaml::ListConsOp>(loc(node), value, list);
      return {consOp};
    });
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genListExpression(const Node node) {
  TRACE();
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  auto listType = mlir::cast<mlir::ocaml::ListType>(*type);
  mlir::Value list = builder.create<mlir::ocaml::UndefOp>(loc(node), listType);
  auto elements = getNamedChildren(node);
  for (auto element : elements) {
    auto elementValue = gen(element);
    if (failed(elementValue)) {
      return failure();
    }
    list = builder.create<mlir::ocaml::ListAppendOp>(loc(node), list, *elementValue);
  }
  return {list};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genFunExpression(const Node node) {
  TRACE();
  return mlirFunctionType(node) |
         and_then([&](auto funType) -> mlir::FailureOr<mlir::ocaml::ClosureType> {
           if (auto closureType = mlir::dyn_cast<mlir::ocaml::ClosureType>(funType)) {
             return closureType;
           }
           return error(node)
                  << "Expected function type for fun expression: " << funType;
         }) |
         and_then([&](auto closureType) -> mlir::FailureOr<mlir::Value> {
           auto anonName = getUniqueName("funexpr");
           VariableScope scope(variables);
           InsertionGuard guard(builder);
           auto letOp =
               builder.create<mlir::ocaml::LetOp>(loc(node), closureType, anonName);
           mlir::Block &bodyBlock = letOp.getBody().emplaceBlock();
           builder.setInsertionPointToStart(&bodyBlock);
           auto parameters = getNamedChildren(node, {"parameter"});
           for (auto [i, parameter] : llvm::enumerate(parameters)) {
             auto paramType = mlirType(parameter);
             if (failed(paramType)) {
               return error(parameter) << "Failed to generate parameter type";
             }
             auto parameterName = getIdentifierTextFromPattern(parameter);
             auto ba = bodyBlock.addArgument(paramType.value(), loc(parameter));
             auto decl = declareVariable(parameterName, ba, loc(parameter));
             if (failed(decl)) {
               return error(parameter) << "Failed to declare parameter";
             }
           }
           auto body = node.getChildByFieldName("body");
           auto bodyValue = gen(body);
           if (failed(bodyValue)) {
             return error(body) << "Failed to generate body";
           }
           builder.create<mlir::ocaml::YieldOp>(loc(node), *bodyValue);
           return {letOp};
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genArrayGetExpression(const Node node) {
  TRACE();
  auto arrayNode = node.getChildByFieldName("array");
  auto indexNode = node.getChildByFieldName("index");
  auto arrayValue = gen(arrayNode);
  if (failed(arrayValue)) {
    return error(node) << "Failed to generate array";
  }
  auto indexValue = gen(indexNode);
  if (failed(indexValue)) {
    return error(node) << "Failed to generate index";
  }
  return builder.createArrayGet(loc(node), *arrayValue, *indexValue);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genIfExpression(const Node node) {
  TRACE();
  InsertionGuard guard(builder);
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  std::optional<mlir::Type> type = *maybeType;
  auto conditionNode = node.getChildByFieldName("condition");
  auto thenNodes = getNamedChildren(node, {"then_clause"});
  assert(thenNodes.size() == 1);
  auto thenNode = thenNodes[0];
  auto elseNodes = getNamedChildren(node, {"else_clause"});
  auto elseNode =
      elseNodes.empty() ? std::nullopt : std::optional<Node>(elseNodes[0]);
  assert(elseNodes.size() <= 1);
  if (elseNodes.empty()) {
    assert(llvm::isa<mlir::ocaml::UnitType>(*type));
  }
  if (llvm::isa<mlir::ocaml::UnitType>(*type)) {
    type = std::nullopt;
  }
  auto maybeConditionValue = gen(conditionNode);
  if (failed(maybeConditionValue)) {
    return failure();
  }
  auto conditionValue = builder.createConvert(
      loc(conditionNode), builder.getI1Type(), *maybeConditionValue);
  SmallVector<mlir::Type> resultType;
  if (type) {
    resultType.push_back(*type);
  }

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc(node), resultType, conditionValue, true, elseNode.has_value());

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto thenValue = gen(thenNode);
  if (failed(thenValue)) {
    return failure();
  }

  if (!resultType.empty()) {
    builder.create<mlir::scf::YieldOp>(loc(node), *thenValue);
  }

  if (elseNode) {
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto elseValue = gen(elseNode.value());
    if (failed(elseValue)) {
      return failure();
    }
    builder.create<mlir::scf::YieldOp>(loc(node), *elseValue);
  }

  return {resultType.empty() ? builder.createUnit(loc(node))
                             : ifOp.getResult(0)};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genString(const Node node) {
  TRACE();
  auto str = getTextStripQuotes(node);
  return builder.createString(loc(node), str);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genSequenceExpression(const Node node) {
  TRACE();
  auto children = getNamedChildren(node);
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  mlir::Value result;
  for (auto child : children) {
    auto maybeValue = gen(child);
    if (failed(maybeValue)) {
      return failure();
    }
    result = *maybeValue;
  }
  return result;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genValuePattern(const Node node) {
  TRACE();
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  auto type = *maybeType;
  if (mlir::isa<mlir::ocaml::UnitType>(type)) {
    return builder.createUnit(loc(node));
  }

  auto placeholder = builder.createPatternVariable(loc(node), type);

  if (failed(declareVariable(node, placeholder, loc(node)))) {
    return failure();
  }

  return {placeholder};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genConstructorPattern(const Node node) {
  TRACE();
  auto ctor = node.getNamedChild(0);
  auto maybeConstructorPath = genConstructorPath(ctor);
  if (failed(maybeConstructorPath)) {
    return error(node) << "Failed to generate constructor path";
  }
  auto constructorPath = *maybeConstructorPath;
  auto value = std::get<mlir::Value>(constructorPath);
  if (!mlir::isa<mlir::ocaml::ClosureType>(value.getType())) {
    return value;
  }
  auto constructorArgNodes =
      SmallVector<Node>(llvm::drop_begin(getNamedChildren(node)));
  auto maybeConstructorArgs = llvm::to_vector(llvm::map_range(
      constructorArgNodes, [this](const Node &arg) { return gen(arg); }));
  if (llvm::any_of(maybeConstructorArgs, failed)) {
    return error(node) << "Failed to generate constructor arguments";
  }
  auto constructorArgs = llvm::map_to_vector(maybeConstructorArgs,
                                             [](auto value) { return *value; });
  auto constructorCall =
      builder.create<mlir::ocaml::CallOp>(loc(node), value, constructorArgs);
  return {constructorCall};
}

llvm::StringRef MLIRGen3::getText(const Node node) {
  return unifier.getTextSaved(node);
}

mlir::FailureOr<mlir::Value>
MLIRGen3::declareVariable(Node node, mlir::Value value, mlir::Location loc, VariableScope *scope) {
  TRACE();
  auto str = getText(node);
  if (node.getType() == "typed_pattern") {
    node = node.getChildByFieldName("pattern");
    str = getText(node);
  }
  DBGS("declaring variable: " << str << ' ' << node.getType() << "\n");
  return declareVariable(str, value, loc, scope);
}

mlir::FailureOr<mlir::Value> MLIRGen3::declareVariable(llvm::StringRef name,
                                                       mlir::Value value,
                                                       mlir::Location loc,
                                                       VariableScope *scope) {
  auto savedName = stringArena.save(name);
  DBGS("declaring '" << savedName << "' of type " << value.getType() << "\n");
  // TODO: handle finalizing the module type later
  // if (shouldAddToModuleType(value.getDefiningOp())) {
  //   getCurrentModuleType().addType(savedName, value.getType());
  // }
  if (scope == nullptr) {
    variables.insert(savedName, value);
  } else {
    variables.insertIntoScope(scope, savedName, value);
  }
  return value;
}

mlir::FailureOr<mlir::Value> MLIRGen3::gen(const Node node) {
  TRACE();
  DBG(unifier.show(node.getCursor()));
  auto type = node.getType();
  InsertionGuard guard(builder);
  static constexpr std::string_view passthroughTypes[] = {
      "parenthesized_expression",
      "then_clause",
      "else_clause",
      "do_clause",
      "value_definition",
      "expression_item",
      "parenthesized_type",
  };
  static constexpr std::string_view ignoreTypes[] = {
      "comment",
      "type_definition",
  };
  if (llvm::is_contained(passthroughTypes, type)) {
    return gen(node.getNamedChild(0));
  } else if (type == "compilation_unit") {
    return genCompilationUnit(node);
  } else if (type == "let_binding") {
    return genLetBinding(node);
  } else if (llvm::is_contained(ignoreTypes, type)) {
    return mlir::Value();
  } else if (type == "number") {
    return genNumber(node);
  } else if (type == "for_expression") {
    return genForExpression(node);
  } else if (type == "application_expression") {
    return genApplicationExpression(node);
  } else if (type == "value_path") {
    return getVariable(node);
  } else if (type == "infix_expression") {
    return genInfixExpression(node);
  } else if (type == "unit") {
    return builder.createUnit(loc(node));
  } else if (type == "let_expression") {
    return genLetExpression(node);
  } else if (type == "value_name") {
    return getVariable(node);
  } else if (type == "match_expression") {
    return genMatchExpression(node);
  } else if (type == "constructor_path") {
    return genConstructorPath(node) |
           and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
             return std::visit(
                 Overload{
                     [&](mlir::func::FuncOp funcOp)
                         -> mlir::FailureOr<mlir::Value> {
                       return {builder.create<mlir::func::ConstantOp>(
                           loc(node), funcOp.getFunctionType(),
                           funcOp.getSymName())};
                     },
                     [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
                       return value;
                     },
                     [&](BuiltinBuilder builder) -> mlir::FailureOr<mlir::Value> {
                       assert(false && "wtf?");
                       return failure();
                     }},
                 callee);
           });
  } else if (type == "constructor_pattern") {
    return genConstructorPattern(node);
  } else if (type == "value_pattern") {
    return genValuePattern(node);
  } else if (type == "sequence_expression") {
    return genSequenceExpression(node);
  } else if (type == "string") {
    return genString(node);
  } else if (type == "array_expression") {
    return genArrayExpression(node);
  } else if (type == "if_expression") {
    return genIfExpression(node);
  } else if (type == "array_get_expression") {
    return genArrayGetExpression(node);
  } else if (type == "fun_expression") {
    return genFunExpression(node);
  } else if (type == "list_expression") {
    return genListExpression(node);
  } else if (type == "cons_expression") {
    return genConsExpression(node);
  } else if (type == "prefix_expression") {
    return genPrefixExpression(node);
  } else if (type == "external") {
    return genExternal(node);
  } else if (type == "module_definition") {
    return genModuleDefinition(node);
  } else if (type == "structure") {
    return genModuleStructure(node);
  } else if (type == "module_type_definition") {
    return mlir::Value(); // not needed in the IR, just for type checking
  } else if (type == "record_expression") {
    return genRecordExpression(node);
  } else if (type == "record_pattern") {
    return genRecordPattern(node);
  } else if (type == "field_get_expression") {
    return genFieldGetExpression(node);
  }
  error(node) << "NYI: " << type << " (" << __LINE__ << ')';
  assert(false);
  return failure();
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ocaml::ModuleOp>> MLIRGen3::gen() {
  TRACE();
  module = builder.create<mlir::ocaml::ModuleOp>(loc(root), unifier.getLastModule());
  pushModule(*module);
  pushModuleType(module->getModuleType());
  module->getBody().emplaceBlock();
  builder.setInsertionPointToStart(module->getBodyBlock());
  auto res = gen(root);
  if (mlir::failed(res)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    llvm::errs() << *module << "\n";
    return mlir::failure();
  }

  if (mlir::failed(module->verify())) {
    DBGS("Failed to verify MLIR for compilation unit\n");
    llvm::errs() << *module << "\n";
    return mlir::failure();
  }
  getCurrentModuleType().finalize();
  popModuleType();

  return std::move(module);
}
