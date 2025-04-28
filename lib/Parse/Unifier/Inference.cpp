#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <llvm/Support/FileSystem.h>

#define DEBUG_TYPE "inference"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"
#include "PathUtilities.h"

namespace ocamlc2 {
using namespace std::string_literals;

namespace { 
static bool shouldSkip(Node node) {
  static constexpr std::string_view shouldSkip[] = {
      "comment",
      "line_number_directive",
      ";;",
  };
  return llvm::any_of(shouldSkip, [&](auto s) { return node.getType() == s; });
}
} // namespace

TypeExpr* Unifier::infer(ts::Cursor cursor) {
  if (shouldSkip(cursor.getCurrentNode())) {
    if (!cursor.gotoNextSibling()) {
      return nullptr;
    }
  }
  DBGS("Inferring type for: " << cursor.getCurrentNode().getType() << '\n');
  DBG(show(cursor.copy(), false));
  auto *te = inferType(cursor.copy());
  ORNULL(te);
  RNULL_IF(anyFatalErrors(), "Failed to infer type");
  DBGS("Inferred type:\n");
  setType(cursor.getCurrentNode(), te);
  maybeDumpTypes(cursor.getCurrentNode(), te);
  DBG(show(cursor.copy(), true));
  return te;
}

TypeExpr* Unifier::inferOpenModule(Cursor ast) {
  TRACE();
  auto name = ast.getCurrentNode().getNamedChild(0);
  auto path = getPathParts(name);
  pushModuleSearchPath(hashPath(path));
  auto currentModulePath = currentModule;
  std::copy(path.begin(), path.end(), std::back_inserter(currentModulePath));
  pushModuleSearchPath(hashPath(currentModulePath));
  return getUnitType();
}

SmallVector<Node> Unifier::flattenType(std::string_view nodeType, Node node) {
  TRACE();
  if (node.getType() != nodeType) {
    return {node};
  }
  SmallVector<Node> nodes;
  auto leftChild = node.getNamedChild(0);
  nodes.push_back(leftChild);
  auto rightChild = node.getNamedChild(1);
  auto flattenedRightChild = flattenFunctionType(rightChild);
  std::copy(flattenedRightChild.begin(), flattenedRightChild.end(), std::back_inserter(nodes));
  DBG(
    DBGS("Flattened function type:\n");
    for (auto [i, n] : llvm::enumerate(nodes)) {
      DBGS(i << ": " << n.getType() << '\n');
    }
  );
  return nodes;
}

FailureOr<ParameterDescriptor> Unifier::describeFunctionArgumentType(Node node) {
  static constexpr std::string_view passthroughTypes[] = {
      "type_variable", "constructed_type", "type_constructor_path",
      "parenthesized_type", "tuple_type", };
  if (llvm::is_contained(passthroughTypes, node.getType())) {
    return ParameterDescriptor{};
  } else if (node.getType() == "labeled_argument_type") {
    const auto optional = node.getChild(0).getType() == "?";
    const auto label = node.getNamedChild(0);
    const auto typeNode = node.getChildByFieldName("type");
    const auto labelKind = optional ? ParameterDescriptor::LabelKind::Optional
                                    : ParameterDescriptor::LabelKind::Labeled;
    return ParameterDescriptor(labelKind, getTextSaved(label), typeNode, std::nullopt);
  }
  FAIL(SSWRAP("Unknown function argument type: " << node.getType()), node);
}

TypeExpr* Unifier::inferLabeledArgumentType(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "labeled_argument_type");
  return infer(node.getChildByFieldName("type"));
}

TypeExpr* Unifier::inferFunctionType(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "function_type");
  auto leftChild = node.getNamedChild(0);
  SmallVector<TypeExpr*> types = {infer(leftChild)};
  ORNULL(types.front());
  auto rightChild = node.getNamedChild(1);
  auto nodes = flattenFunctionType(rightChild);
  SmallVector<ParameterDescriptor> parameterDescriptors;
  for (auto [i, n] : llvm::enumerate(nodes)) {
    auto *childType = infer(n);
    ORNULL(childType);
    types.push_back(childType);
    DBGS(i << ": " << *childType << '\n');
    auto desc = describeFunctionArgumentType(n);
    RNULL_IF(failed(desc), "Failed to describe function argument type");
    parameterDescriptors.push_back(desc.value());
  }
  return getFunctionType(types, parameterDescriptors);
}

TypeExpr* Unifier::inferConstructorPattern(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_pattern");
  auto *ctorType = [&] {
    auto name = node.getNamedChild(0);
    auto *ctorType = inferConstructorPath(name.getCursor());
    auto *ctorTypeOperator = llvm::dyn_cast<TypeOperator>(ctorType);
    assert(ctorTypeOperator && "Expected constructor type operator");
    return ctorTypeOperator;
  }();
  ORNULL(ctorType);
  auto declaredArgs = SmallVector<TypeExpr*>(ctorType->getArgs());
  if (declaredArgs.empty()) {
    // If we get an empty type operator, we're matching against a constructor
    // with no arguments, so we can just return the type operator directly
    // and there's nothing to unify.
    return ctorType;
  }
  // Otherwise, we have a function mapping the variant constructor arguments to the variant type
  // itself, so we drop the return type to get the pattern arguments to match against.
  auto *varaintType = declaredArgs.pop_back_val();
  auto argsType = [&] {
    auto ctorArgumentsPattern = node.getNamedChild(1);
    auto *argsType = infer(ctorArgumentsPattern);
    return argsType;
  }();
  if (declaredArgs.size() == 1) {
    // If we have a single argument, matches will be directly against it and not wrapped
    // in a tuple type.
    UNIFY_OR_RNULL(argsType, declaredArgs[0]);
    return varaintType;
  } else {
    auto args = llvm::cast<TypeOperator>(argsType)->getArgs();
    assert(args.size() == declaredArgs.size() && "Expected constructor pattern to have the same number of "
                      "arguments as the constructor type operator");
    for (auto [arg, declaredArg] : llvm::zip(args, declaredArgs)) {
      UNIFY_OR_RNULL(arg, declaredArg);
    }
    return varaintType;
  }
}

TypeExpr* Unifier::inferTupleExpression(Cursor ast) {
  auto types = llvm::map_to_vector(getNamedChildren(ast.getCurrentNode()), [&](Node n) {
    return infer(n);
  });
  return getTupleType(types);
}

TypeExpr* Unifier::inferTupleType(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "tuple_type");
  auto nodes = flattenTupleType(node);
  SmallVector<TypeExpr*> types;
  for (auto n : nodes) {
    types.push_back(infer(n));
  }
  return getTupleType(types);
}

TypeExpr* Unifier::inferTuplePattern(ts::Node node) {
  SmallVector<TypeExpr*> types;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto *childType = infer(node.getNamedChild(i));
    if (auto *childTupleType = llvm::dyn_cast<TupleOperator>(childType)) {
      std::copy(childTupleType->getArgs().begin(),
                childTupleType->getArgs().end(), std::back_inserter(types));
    } else {
      types.push_back(childType);
    }
  }
  return getTupleType(types);
}

TypeExpr* Unifier::inferPattern(ts::Node node) {
  static constexpr std::string_view passthroughPatterns[] = {
    "number", "string",
  };
  if (node.getType() == "value_pattern") {
    return inferValuePattern(node.getCursor());
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(node.getCursor());
  } else if (node.getType() == "parenthesized_pattern") {
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "constructor_pattern") {
    return inferConstructorPattern(node.getCursor());
  } else if (node.getType() == "tuple_pattern") {
    return inferTuplePattern(node);
  } else if (node.getType() == "record_pattern") {
    return inferRecordPattern(node.getCursor());
  } else if (llvm::is_contained(passthroughPatterns, node.getType())) {
    return infer(node);
  }
  show(node.getCursor(), true);
  DBGS("Unknown pattern type: " << node.getType() << '\n');
  assert(false && "Unknown pattern type");
  return nullptr;
}

TypeExpr* Unifier::inferMatchCase(TypeExpr* matcheeType, ts::Node node) {
  // Declared variables are only in scope during the body of the match case.
  detail::Scope scope(this);
  assert(node.getType() == "match_case");
  const auto namedChildren = node.getNumNamedChildren();
  const auto hasGuard = namedChildren == 3;
  const auto bodyNode = hasGuard ? node.getNamedChild(2) : node.getNamedChild(1);
  auto *matchCaseType = infer(node.getNamedChild(0));
  UNIFY_OR_RNULL(matchCaseType, matcheeType);
  if (hasGuard) {
    auto *guardType = infer(node.getNamedChild(1));
    UNIFY_OR_RNULL(guardType, getBoolType());
  }
  auto *bodyType = infer(bodyNode);
  setType(bodyNode, bodyType);
  return bodyType;
}

TypeExpr *Unifier::inferMatchExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "match_expression");
  auto matchee = node.getNamedChild(0);
  const auto namedChildren = node.getNumNamedChildren();
  if (matchee.getType() == "match_case") {
    RNULL("NYI: match expression with match case");
    // If we don't get a match-case, we're implicitly creating
    // a function with the matchee not available as a symbol yet.
    auto *matcheeType = createTypeVariable();
    declareVariable(node, matcheeType);
    SmallVector<TypeExpr*> functionType = {matcheeType};
    unsigned i = 1;
    while (i < namedChildren) {
      auto *type = inferMatchCase(matcheeType, node.getNamedChild(i++));
      // size .gt. 1, because the implicit matchee is the first element of the case types
      if (functionType.size() > 1) {
        UNIFY_OR_RNULL(type, functionType.back());
      }
      functionType.push_back(type);
    }
    return getFunctionType(functionType);
  } else {
    // Otherwise, we're matching against a value, so we need to infer the type of the
    // matchee.
    auto *matcheeType = infer(matchee);
    TypeExpr *resultType = nullptr;
    unsigned i = 1;
    while (i < namedChildren) {
      auto caseNode = node.getNamedChild(i++);
      auto *type = inferMatchCase(matcheeType, caseNode);
      if (resultType != nullptr) {
        UNIFY_OR_RNULL(type, resultType);
      }
      resultType = type;
    }
    return resultType;
  }
}

TypeExpr *Unifier::declareConcreteVariable(Node node) {
  auto *type = createTypeVariable();
  concreteTypes.insert(type);
  return declareVariable(node, type);
}

ParameterDescriptor Unifier::describeParameter(Node node) {
  TRACE();
  assert(node.getType() == "parameter");
  const auto children = getChildren(node);
  auto pattern = [&] {
    const auto pattern = node.getChildByFieldName("pattern");
    assert(!pattern.isNull() && "Expected pattern for parameter");
    return (pattern.getType() == "unit") ? std::nullopt : toOptional(pattern);
  }();
  const auto prefix = node.getChild(0);
  const bool isOptional = prefix.getType() == "?";
  const bool isLabeled = prefix.getType() == "~" or isOptional;
  auto type = toOptional(node.getChildByFieldName("type"));
  if (pattern.has_value() && pattern->getType() == "typed_pattern") {
    type = toOptional(pattern->getChildByFieldName("type"));
    pattern = toOptional(pattern->getChildByFieldName("pattern"));
  }
  const auto defaultValue = toOptional(node.getChildByFieldName("default"));
  const auto *labelIter = llvm::find_if(children, [](Node n) {
    return n.getType() == "label_name";
  });
  const auto label = [&] -> std::optional<llvm::StringRef> {
    if (labelIter != children.end()) {
      DBGS("Describing labeled parameter\n");
      return getTextSaved(*labelIter);
    } else if (isOptional) {
      DBGS("Describing optional parameter\n");
      assert(pattern && "Expected pattern for optional parameter");
      return getTextSaved(*pattern);
    } else if (isLabeled) {
      DBGS("Describing labeled parameter\n");
      return getTextSaved(*pattern);
    }
    return std::nullopt;
  }();
  const auto labelKind = isOptional  ? ParameterDescriptor::LabelKind::Optional
                         : isLabeled ? ParameterDescriptor::LabelKind::Labeled
                                     : ParameterDescriptor::LabelKind::None;
  const auto desc =
      ParameterDescriptor{labelKind, label, type, defaultValue};
  return desc;
}

SmallVector<ParameterDescriptor>
Unifier::describeParameters(SmallVector<Node> parameters) {
  auto descs = llvm::map_to_vector(parameters, [&](Node n) -> ParameterDescriptor {
    return describeParameter(n);
  });
  assert(descs.back().labelKind != ParameterDescriptor::LabelKind::Optional &&
         "Expected last parameter to be non-optional");
  return descs;
}

TypeExpr *Unifier::declareFunctionParameter(ParameterDescriptor desc, Node node) {
  DBGS(node.getSExpr().get() << '\n');
  auto pattern = node.getChildByFieldName("pattern");
  if (pattern.getType() == "unit") {
    return getUnitType();
  } else if (pattern.getType() == "typed_pattern") {
    TRACE();
    auto *type = infer(desc.type.value());
    pattern = pattern.getChildByFieldName("pattern");
    setType(pattern, type);
    return declareVariable(pattern, type);
  } else {
    TRACE();
    TypeExpr *type;
    if (desc.type.has_value()) {
      TRACE();
      type = infer(desc.type.value());
    } else {
      TRACE();
      auto *tv = createTypeVariable();
      concreteTypes.insert(tv);
      type = tv;
    }
    if (desc.isOptional() && not desc.hasDefaultValue()) {
      TRACE();
      type = getOptionalTypeOf(type);
    }
    setType(pattern, type);
    return declareVariable(pattern, type);
  }
}

TypeExpr *Unifier::declareFunctionParameter(Node node) {
  assert(node.getType() == "parameter");
  node = node.getNamedChild(0);
  if (node.getType() == "unit") {
    return getUnitType();
  } else if (node.getType() == "typed_pattern") {
    auto name = node.getNamedChild(0);
    auto *type = infer(node.getNamedChild(1));
    setType(node, type);
    return declareVariable(name, type);
  } else {
    auto *type = createTypeVariable();
    concreteTypes.insert(type);
    setType(node, type);
    return declareVariable(node, type);
  }
}

TypeExpr *Unifier::inferLetBindingFunction(Node name, SmallVector<Node> parameters, Node body) {
  DBGS("non-recursive let binding, inferring body type\n");
  auto parameterDescriptors = describeParameters(parameters);
  auto [returnType, types] = [&]() { 
    detail::Scope scope(this);
    SmallVector<TypeExpr*> types = llvm::map_to_vector(llvm::zip(parameterDescriptors, parameters), [&](auto arg) -> TypeExpr* {
      auto [desc, param] = arg;
      return declareFunctionParameter(desc, param);
    });
    auto *returnType = infer(body);
    return std::make_pair(returnType, types);
  }();
  ORNULL(returnType);
  types.push_back(returnType);
  auto *funcType = getFunctionType(types, parameterDescriptors);
  ORNULL(funcType);
  declareVariable(name, funcType);
  return funcType;
}

TypeExpr *Unifier::inferLetBindingRecursiveFunction(Node name, SmallVector<Node> parameters, Node body) {
  DBGS("recursive let binding, declaring function type before body\n");
  auto *tv = createTypeVariable();
  declareVariable(name, tv);
  auto *funcType = [&]() -> TypeExpr* {
    detail::Scope scope(this);
    SmallVector<TypeExpr*> types = llvm::map_to_vector(parameters, [&](Node n) -> TypeExpr* {
      return declareFunctionParameter(n);
    });
    auto *bodyType = infer(body);
    types.push_back(bodyType);
    return getFunctionType(types);
  }();
  ORNULL(funcType);
  RNULL_IF(failed(unify(funcType, tv)), "Failed to unify function type with type variable");
  return funcType;
}

TypeExpr *Unifier::inferLetBindingValue(Node name, Node body) {
  DBGS("variable let binding, no parameters\n");
  auto *bodyType = infer(body);
  ORNULL(bodyType);
  declareVariable(name, bodyType);
  return bodyType;
}


TypeExpr* Unifier::inferLetBinding(Cursor ast) {
  TRACE();
  TypeVarEnvScope scope(typeVarEnv);
  auto node = ast.getCurrentNode();
  const bool isRecursive = isLetBindingRecursive(ast.copy());
  auto name = node.getChildByFieldName("pattern");
  auto body = node.getChildByFieldName("body");
  assert(!name.isNull() && !body.isNull() && "Expected name and body");
  const auto parameters = llvm::filter_to_vector(getNamedChildren(node), [](Node n) {
    return n.getType() == "parameter";
  });
  auto declaredReturnTypeNode = node.getChildByFieldName("type");
  TypeExpr *declaredReturnType =
      declaredReturnTypeNode.isNull() ? nullptr : infer(declaredReturnTypeNode);
  TypeExpr* inferredReturnType = nullptr;
  if (parameters.empty()) {
    assert(!isRecursive && "recursive let binding with no parameters");
    inferredReturnType = inferLetBindingValue(name, body);
    ORNULL(inferredReturnType);
    if (declaredReturnType != nullptr) {
      RNULL_IF(
          failed(unify(inferredReturnType, declaredReturnType)),
          "Failed to unify inferred return type with declared return type");
    }
    return inferredReturnType;
  } else {
    if (isRecursive) {
      inferredReturnType =
          inferLetBindingRecursiveFunction(name, parameters, body);
    } else {
      inferredReturnType = inferLetBindingFunction(name, parameters, body);
    }
    if (declaredReturnType != nullptr) {
      SmallVector<TypeExpr*> typeVars = llvm::map_to_vector(parameters, [&] (auto) -> TypeExpr* { return createTypeVariable(); });
      typeVars.push_back(declaredReturnType);
      auto *expectedType = getFunctionType(typeVars);
      UNIFY_OR_RNULL(inferredReturnType, expectedType);
    }
    return inferredReturnType;
  }
}

TypeExpr* Unifier::inferIfExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  auto childCount = node.getNumNamedChildren();
  DBGS("ternary with " << childCount << " children\n");
  auto *condition = infer(node.getNamedChild(0));
  UNIFY_OR_RNULL(condition, getBoolType());
  switch (childCount) {
    case 3: {
      auto *thenBranch = infer(node.getNamedChild(1));
      auto *elseBranch = infer(node.getNamedChild(2));
      UNIFY_OR_RNULL(thenBranch, elseBranch);
      return thenBranch;
    }
    case 2: {
      auto *thenBranch = infer(node.getNamedChild(0));
      UNIFY_OR_RNULL(getUnitType(), thenBranch);
      return thenBranch;
    }
    default: {
      assert(false && "Expected 2 or 3 children for if expression");
      return nullptr;
    }
  }
}

TypeExpr* Unifier::declareVariable(Node node, TypeExpr* type) {
  if (node.getType() == "parenthesized_operator") {
    node = node.getNamedChild(0);
  }
  declareVariable(getTextSaved(node), type);
  setType(node, type);
  return type;
}

TypeExpr* Unifier::inferForExpression(Cursor ast) {
  assert(ast.gotoFirstChild());
  assert(ast.gotoNextSibling());
  auto id = ast.getCurrentNode();
  declareVariable(id, getIntType());
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "=");
  assert(ast.gotoNextSibling());
  auto firstBound = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "to" or ast.getCurrentNode().getType() == "downto");
  assert(ast.gotoNextSibling());
  auto secondBound = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "do_clause");
  auto body = ast.getCurrentNode();
  assert(!ast.gotoNextSibling());
  UNIFY_OR_RNULL(infer(firstBound), getIntType());
  UNIFY_OR_RNULL(infer(secondBound), getIntType());
  (void)infer(body);
  return getUnitType();
}

TypeExpr* Unifier::inferCompilationUnit(Cursor ast) {
  TRACE();
  const auto currentModule = filePathToModuleName(sources.back().filepath);
  pushModule(stringArena.save(currentModule));
  auto *t = getUnitType();
  if (ast.gotoFirstChild()) {
    do {
      if (!shouldSkip(ast.getCurrentNode())) {
        t = infer(ast.copy());
      }
    } while (ast.gotoNextSibling());
  }
  popModule();
  return t;
}

TypeExpr* Unifier::inferValuePath(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_path" or node.getType() == "module_path");
  auto pathParts = getPathParts(node);
  return getVariableType(pathParts);
}

std::pair<FunctionOperator *, TypeExpr *>
Unifier::normalizeFunctionType(TypeExpr *declaredType,
                               SmallVector<Node> arguments) {
  TRACE();
  auto *declaredFuncType = llvm::dyn_cast<FunctionOperator>(declaredType);
  if (!declaredFuncType) {
    DBGS("Not a function type: " << *declaredType << '\n');
    // If we don't have a function operator then we can't know anything about the callee
    // and there's nothing we can normalize at this point.
    auto inferredTypes = llvm::map_to_vector(arguments, [&](auto arg) {
      return infer(arg);
    });
    inferredTypes.push_back(createTypeVariable());
    auto *inferredType = getFunctionType(inferredTypes);
    return std::make_pair(inferredType, declaredType);
  }
  DBGS("Normalizing function type: " << *declaredFuncType << " with arguments:\n");
  DBG(llvm::for_each(arguments, [&](auto arg) {
    DBGS(arg.getType() << ' ' << getText(arg) << '\n');
  }));
  auto descs = declaredFuncType->parameterDescriptors;
  auto functionTypeArgs = declaredFuncType->getArgs();
  const auto isVarargs = declaredFuncType->isVarargs();
  DBGS("isVarargs: " << isVarargs << '\n');
  auto [actualLabeled, actualPositional] = getLabeledArguments(arguments);
  SmallVector<TypeExpr*> normalizedArgumentTypes;
  SmallVector<TypeExpr*> missingDeclaredArguments; // for partial application
  SmallVector<TypeExpr*> passedDeclaredArguments; // for partial application
  SmallVector<ParameterDescriptor> missingDescs;
  SmallVector<ParameterDescriptor> passedDescs;
  auto it = actualPositional.begin();
  for (auto [i, desc] : enumerate(descs)) {
    DBGS("Normalizing argument " << i << " with descriptor " << desc << '\n');
    auto *declaredArgType = functionTypeArgs[i];
    if (desc.isPositional()) {
      if (it == actualPositional.end()) {
        if (llvm::isa<VarargsOperator>(declaredArgType)) {
          DBGS("WARNING: fragile handling of varargs.\n");
          continue;
        }
        missingDeclaredArguments.push_back(declaredArgType);
        missingDescs.push_back(desc);
      } else {
        auto arg = *it++;
        auto *type = infer(arg);
        normalizedArgumentTypes.push_back(type);
        setType(arg, type);
        passedDeclaredArguments.push_back(declaredArgType);
        passedDescs.push_back(desc);
      }
    } else {
      assert(!isVarargs && "Varargs should not have labeled arguments");
      auto label = desc.label;
      assert(label.has_value() && "Expected labeled argument");
      auto actual = llvm::find_if(
          actualLabeled, [&](auto pair) { return pair.first == label; });
      if (actual == actualLabeled.end()) {
        if (desc.isOptional()) {
          if (desc.defaultValue) {
            normalizedArgumentTypes.push_back(infer(*desc.defaultValue));
            passedDeclaredArguments.push_back(declaredArgType);
            passedDescs.push_back(desc);
          } else {
            auto *Optional = getOptionalType();
            normalizedArgumentTypes.push_back(Optional);
            passedDeclaredArguments.push_back(Optional);
            passedDescs.push_back(desc);
          }
        } else {
          missingDeclaredArguments.push_back(declaredArgType);
          missingDescs.push_back(desc);
        }
      } else {
        auto arg = actual->second;
        auto *type = infer(arg);
        if (desc.isOptional() and not desc.hasDefaultValue()) {
          type = getOptionalTypeOf(type);
        }
        setType(arg, type);
        normalizedArgumentTypes.push_back(type);
        passedDeclaredArguments.push_back(declaredArgType);
        passedDescs.push_back(desc);
        actualLabeled.erase(actual);
      }
    }
  }
  assert(missingDeclaredArguments.size() == missingDescs.size());
  assert(passedDeclaredArguments.size() == passedDescs.size());
  assert(actualLabeled.empty() && "Expected all labeled arguments to be consumed");

  normalizedArgumentTypes.push_back(createTypeVariable());
  auto *normalizedInferredFuncType = getFunctionType(normalizedArgumentTypes, passedDescs);

  auto *reDeclaredFuncType = [&] {
    if (missingDeclaredArguments.empty()) {
      DBGS("No missing arguments, returning original function type\n");
      // If there were no missing arguments then the declared function type is already normalized
      return declaredFuncType;
    }
    if (missingDeclaredArguments.size() == 1 && declaredFuncType->isVarargs()) {
      DBGS("WARNING: fragile handling of varargs.\n");
      return declaredFuncType;
    }

    DBGS("Missed " << missingDeclaredArguments.size() << " arguments\n");

    // Otherwise, collect the passed declared arguments and create a new function type
    // with those arguments and returning a function type taking the remaining arguments.
    auto remainingDeclaredArguments = missingDeclaredArguments;
    auto *declaredReturnType = declaredFuncType->back();
    remainingDeclaredArguments.push_back(declaredReturnType);
    auto *partiallyAppliedFuncType = getFunctionType(remainingDeclaredArguments, missingDescs);
    DBGS("Partially applied function type: " << *partiallyAppliedFuncType << '\n');
    passedDeclaredArguments.push_back(partiallyAppliedFuncType);
    auto *curriedInferredFuncType = getFunctionType(passedDeclaredArguments, passedDescs);
    DBGS("Curried inferred function type: " << *curriedInferredFuncType << '\n');
    return curriedInferredFuncType;
  }();
  return {normalizedInferredFuncType, reDeclaredFuncType};
}

std::pair<SmallVector<std::pair<llvm::StringRef, Node>>, std::set<Node>>
Unifier::getLabeledArguments(SmallVector<Node> arguments) {
  TRACE();
  SmallVector<std::pair<llvm::StringRef, Node>> labeledArguments;
  std::set<Node> positionalArguments;
  for (auto arg : arguments) {
    if (arg.getType() == "labeled_argument") {
      labeledArguments.emplace_back(getTextSaved(arg.getNamedChild(0)),
                                    arg.getChildByFieldName("expression"));
    } else {
      positionalArguments.insert(arg);
    }
  }
  return std::make_pair(std::move(labeledArguments), std::move(positionalArguments));
}

TypeExpr* Unifier::inferApplicationExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "application_expression");
  const auto function = node.getChildByFieldName("function");
  auto declaredFuncType = infer(function);
  ORNULL(declaredFuncType);
  auto arguments = getArguments(node);
  DBGS("found " << arguments.size() << " arguments\n");
  auto [normalizedInferredFunctionType, normalizedDeclaredFunctionType] =
      normalizeFunctionType(declaredFuncType, arguments);
  UNIFY_OR_RNULL(normalizedInferredFunctionType, normalizedDeclaredFunctionType);
  return normalizedInferredFunctionType->back();
}

FunctionOperator *Unifier::getFunctionTypeForPartialApplication(FunctionOperator *func, unsigned arity) {
  TRACE();
  auto fullArgs = func->getArgs();
  const auto declaredArity = fullArgs.size() - 1;
  assert(arity <= declaredArity && "Cant partially apply a function with more arguments than it has");

  // TODO: partial application of varargs does not work - we need to inspect the arguments
  // to see what the proper arity is before seeing if an application is partial or not.
  // For now, use the erroneous behavior of only allowing partial application if the
  // varargs is empty.
  const auto isVarargs = func->isVarargs();

  if (arity == declaredArity || (isVarargs && arity == declaredArity - 1)) {
    DBGS("No partial application needed, returning original function type\n");
    return func;
  }
  auto curriedArgs = SmallVector<TypeExpr*>(fullArgs.begin(), fullArgs.begin() + arity);
  auto returnTypeArgs = SmallVector<TypeExpr*>(fullArgs.begin() + arity, fullArgs.end());
  auto *functionReturnType = getFunctionType(returnTypeArgs);
  curriedArgs.push_back(functionReturnType);
  auto *curriedFuncType = getFunctionType(curriedArgs);
  DBGS("Curried function type: " << *curriedFuncType << '\n');
  return curriedFuncType;
}

TypeExpr* Unifier::inferConstructorPath(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_path");
  auto pathParts = getPathParts(node);
  return getVariableType(pathParts);
}

TypeExpr* Unifier::inferArrayExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "array_expression");
  auto *elementType = createTypeVariable();
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    auto *childType = infer(child);
    RNULL_IF(failed(unify(childType, elementType)), "failed to unify");
  }
  return getArrayTypeOf(elementType);
}

TypeExpr* Unifier::inferProductExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "product_expression");
  SmallVector<TypeExpr*> types;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    auto *childType = infer(child);
    if (auto *tupleType = llvm::dyn_cast<TupleOperator>(childType)) {
      std::copy(tupleType->getArgs().begin(), tupleType->getArgs().end(),
                std::back_inserter(types));
    } else {
      types.push_back(childType);
    }
  }
  return getTupleType(types);
}

TypeExpr* Unifier::inferArrayGetExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "array_get_expression");
  auto inferredArrayType = infer(node.getNamedChild(0));
  auto indexType = infer(node.getNamedChild(1));
  RNULL_IF(failed(unify(indexType, getIntType())), "failed to unify");
  auto *arrayType = getArrayType();
  RNULL_IF(failed(unify(arrayType, inferredArrayType)), "failed to unify");
  return arrayType->back();
}

TypeExpr* Unifier::inferInfixExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "infix_expression");
  assert(ast.gotoFirstChild());
  auto lhsType = infer(ast.getCurrentNode());
  ORNULL(lhsType);
  assert(ast.gotoNextSibling());
  auto op = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  auto rhsType = infer(ast.getCurrentNode());
  ORNULL(rhsType);
  assert(!ast.gotoNextSibling());
  auto *opType = getVariableType(op);
  ORNULL(opType);
  auto *funcType = getFunctionType({lhsType, rhsType, createTypeVariable()});
  ORNULL(funcType);
  UNIFY_OR_RNULL(opType, funcType);
  return funcType->back();
}

TypeExpr* Unifier::inferGuard(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "guard");
  auto *type = infer(node.getNamedChild(0));
  ORNULL(type);
  UNIFY_OR_RNULL(type, getBoolType());
  return getBoolType();
}

TypeExpr* Unifier::inferLetExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "let_expression");
  auto children = getNamedChildren(node);
  auto valueExpression = llvm::find_if(children, [](auto n) {
    return n.getType() == "value_definition";
  });
  assert(valueExpression != children.end() && "Expected value expression in let expression");
  auto body = node.getChildByFieldName("body");
  detail::Scope scope(this);
  infer(*valueExpression);
  return infer(body);
}

TypeExpr* Unifier::inferListExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "list_expression");
  SmallVector<TypeExpr*> args;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto *type = infer(node.getNamedChild(i));
    RNULL_IF((not args.empty() && failed(unify(type, args.back()))),
             "Failed to unify list element type with previous element type");
    args.push_back(type);
  }
  return getListTypeOf(args.back());
}

TypeExpr* Unifier::inferFunctionExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "fun_expression");
  SmallVector<TypeExpr*> types;
  detail::Scope scope(this);
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    if (child.getType() == "parameter") {
      auto *type = declareConcreteVariable(child.getNamedChild(0));
      types.push_back(type);
    } else {
      assert(i == node.getNumNamedChildren() - 1 && "Expected body after parameters");
      types.push_back(infer(child));
    }
  }
  return getFunctionType(types);
}

TypeExpr* Unifier::inferSequenceExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  auto *last = getUnitType();
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    last = infer(node.getNamedChild(i));
  }
  return last;
}

TypeExpr* Unifier::inferModuleDefinition(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_definition");
  auto *moduleBinding = inferModuleBinding(node.getNamedChild(0).getCursor());
  ORNULL(moduleBinding);
  setType(node, moduleBinding);
  return moduleBinding;
}

TypeExpr* Unifier::inferModuleBinding(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_binding");
  auto name = node.getNamedChild(0);
  assert(!name.isNull() && "Expected module name");
  assert(name.getType() == "module_name" && "Expected module name");
  detail::ModuleScope ms{*this, getText(name)};
  const auto numChildren = node.getNumNamedChildren();
  std::optional<Node> signature;
  std::optional<Node> structure;
  for (unsigned i = 1; i < numChildren; ++i) {
    auto child = node.getNamedChild(i);
    if (child.getType() == "signature") {
      signature = child;
    } else if (child.getType() == "structure") {
      structure = child;
    } else {
      show(child.getCursor(), true);
      assert(false && "Unknown module binding child type");
    }
  }
  saveInterfaceDecl(SSWRAP("module " << getTextSaved(name) << " : sig"));
  if (signature) {
    inferModuleSignature(signature->getCursor());
  }
  if (structure) {
    inferModuleStructure(structure->getCursor());
  }
  saveInterfaceDecl("end");
  return createTypeOperator(
      hashPath(ArrayRef<StringRef>{"Module", getTextSaved(name)}), {});
}

TypeExpr* Unifier::inferModuleSignature(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "signature");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    infer(child);
  }
  return getUnitType();
}

TypeExpr* Unifier::inferModuleStructure(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "structure");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    infer(child);
  }
  return getUnitType();
}

TypeExpr* Unifier::inferValueSpecification(Cursor ast) {
  TRACE();
  TypeVarEnvScope scope(typeVarEnv);
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_specification");
  auto name = node.getNamedChild(0);
  auto specification = node.getNamedChild(1);
  auto *type = [&] {
    detail::Scope scope(this);
    return infer(specification);
  }();
  declareVariable(name, type);
  return type;
}

FailureOr<std::pair<llvm::StringRef, TypeExpr*>> Unifier::inferFieldPattern(Node node) {
  auto name = getTextSaved(node.getNamedChild(0));
  auto pattern = toOptional(node.getChildByFieldName("pattern"));
  if (name == "_") {
    return {std::make_pair("", getWildcardType())};
  }
  if (!pattern) {
    return {std::make_pair(name, getWildcardType())};
  }
  auto *type = infer(*pattern);
  FAIL_IF(type == nullptr, "Failed to infer field pattern type");
  return {std::make_pair(name, type)};
}

TypeExpr *Unifier::findMatchingRecordType(TypeExpr *type) {
  ERROR("NYI");
  return nullptr;
}

TypeExpr* Unifier::inferRecordPattern(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "record_pattern");
  if (llvm::any_of(getChildren(node), [](Node n) {
    return n.getType() == "_";
  })) {
    assert(false && "Record pattern with unnamed wildcard field pattern is unhandled in tree-sitter grammar.");
    return nullptr;
  }
  auto children = getNamedChildren(node);
  auto fieldPatterns = llvm::map_to_vector(children, [&](Node n) {
    return inferFieldPattern(n);
  });
  RNULL_IF(llvm::any_of(fieldPatterns, [](auto &&p) {
    return failed(p);
  }), "Failed to infer field pattern type");
  const auto fieldNames = llvm::map_to_vector(fieldPatterns, [](auto &&p) {
    return p->first;
  });
  const auto fieldTypes = llvm::map_to_vector(fieldPatterns, [](auto &&p) {
    return p->second;
  });
  return getRecordType(RecordOperator::getAnonRecordName(), fieldTypes, fieldNames);
}

TypeExpr* Unifier::inferRecordExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  auto children = llvm::filter_to_vector(getNamedChildren(node), [](Node n) {
    return n.getType() == "field_expression";
  });
  assert(node.getType() == "record_expression");
  auto fieldExpressions = llvm::map_to_vector(children, [&](Node n) -> TypeExpr* {
    auto *type = infer(n.getChildByFieldName("body"));
    ORNULL(type);
    return setType(n, type);
  });
  RNULL_IF(llvm::any_of(fieldExpressions, [](TypeExpr* type) {
    return type == nullptr;
  }), "Expected record expression to be a record type");
  auto fieldNames = llvm::map_to_vector(children, [&](Node n) {
    auto children = getNamedChildren(n);
    auto name = llvm::find_if(children, [](Node n) {
      return n.getType() == "field_path";
    });
    assert(name != children.end() && "Expected field path");
    return getTextSaved(*name);
  });
  // Just so the record type itself will do the normalization for us
  auto *anonRecordType = getRecordType(RecordOperator::getAnonRecordName(), fieldExpressions, fieldNames);
  ORNULL(anonRecordType);
  auto anonRecordTypes = anonRecordType->getArgs();
  auto anonRecordFieldNames = anonRecordType->getFieldNames();
  for (auto [recordName, seenFieldNames] : seenRecordFields) {
    if (seenFieldNames.size() != fieldExpressions.size()) {
      continue;
    }
    auto *maybeSeenRecordType = getDeclaredType(recordName);
    ORNULL(maybeSeenRecordType);
    auto *seenRecordType = llvm::dyn_cast<RecordOperator>(maybeSeenRecordType);
    RNULL_IF_NULL(seenRecordType, "Expected record type");
    auto seenRecordTypes = seenRecordType->getArgs();

    // If we're able to unify the seen record type with what we know about the record type
    // for the current expression, then we have found a match.
    if (llvm::equal(seenFieldNames, anonRecordFieldNames)) {
      if (llvm::all_of_zip(anonRecordTypes, seenRecordTypes, [this](auto *a, auto *b) {
        return succeeded(unify(a, b));
      })) {
        return seenRecordType;
      }
    }
  }
  return ERROR("No matching record type found", node);
}

TypeExpr* Unifier::inferFieldGetExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "field_get_expression");
  auto record = node.getChildByFieldName("record");
  auto *declaredType = getVariableType(record);
  auto field = getTextSaved(node.getChildByFieldName("field"));
  if (auto *recordOp = llvm::dyn_cast<RecordOperator>(declaredType)) {
    auto fields = recordOp->getFieldNames();
    auto fieldTypeIt = llvm::find(fields, field);
    if (fieldTypeIt == fields.end()) {
      return ERROR(SSWRAP("Field '" << field << "' not found in record type: " << *declaredType), node);
    }
    auto index = std::distance(fields.begin(), fieldTypeIt);
    return recordOp->at(index);
  }
  for (auto [recordName, fieldNames] : seenRecordFields) {
    if (auto it = llvm::find(fieldNames, field); it != fieldNames.end()) {
      auto *recordTypeVar = getDeclaredType(recordName);
      auto *recordType = llvm::dyn_cast<RecordOperator>(recordTypeVar);
      if (not recordType) {
        std::string str;
        llvm::raw_string_ostream ss(str);
        ss << "Type " << *recordTypeVar
           << " was saved as a record type, but was not a record type when it "
              "was retrieved later.";
        RNULL(ss.str(), node);
      }
      auto index = std::distance(fieldNames.begin(), it);
      return recordType->at(index);
    }
  }
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "Unbound record field '" << field << "'";
  RNULL(ss.str(), node);
}

RecordOperator* Unifier::inferRecordDeclaration(llvm::StringRef recordName, Cursor ast) {
  auto *type = inferRecordDeclaration(std::move(ast));
  ORNULL(type);
  auto *recordType = llvm::dyn_cast<RecordOperator>(type);
  ORNULL(recordType);
  auto fieldNames = recordType->getFieldNames();
  DBGS("Recording field names for record type: " << recordName << '\n');
  seenRecordFields.emplace_back(recordName, fieldNames);
  return getRecordType(recordName, recordType->getArgs(), fieldNames);
}

RecordOperator* Unifier::inferRecordDeclaration(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "record_declaration");
  SmallVector<llvm::StringRef> fieldNames;
  SmallVector<TypeExpr*> fieldTypes;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    assert(child.getType() == "field_declaration");
    auto name = child.getNamedChild(0);
    auto type = infer(child.getNamedChild(1));
    auto text = getTextSaved(name);
    if (llvm::find(fieldNames, text) != fieldNames.end()) {
      RNULL("Duplicate field name", name);
    }
    fieldNames.push_back(text);
    fieldTypes.push_back(type);
  }
  return getRecordType("<anon>", fieldTypes, fieldNames);
}

TypeExpr* Unifier::inferTypeConstructorPath(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_constructor_path");
  return getDeclaredType(node);
}

TypeExpr* Unifier::inferTypeDefinition(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_definition");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    auto *type = inferTypeBinding(child.getCursor());
    ORNULL(type);
    setType(child, type);
  }
  return getUnitType();
}

TypeExpr* Unifier::inferVariantConstructor(VariantOperator* variantType, Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_declaration");
  auto name = node.getNamedChild(0);
  if (node.getNumNamedChildren() == 1) {
    variantType->addConstructor(getTextSaved(name));
    return declareVariable(name, variantType);
  }
  auto parameters = [&] {
    SmallVector<TypeExpr*> types;
    for (unsigned i = 1; i < node.getNumNamedChildren(); ++i) {
      types.push_back(infer(node.getNamedChild(i)));
    }
    return types;
  }();
  auto *functionType = [&] {
    if (parameters.size() == 1) {
      parameters.push_back(variantType);
      return getFunctionType(parameters);
    }
    auto *tupleType = getTupleType(parameters);
    return getFunctionType({tupleType, variantType});
  }();
  declareVariable(name, functionType);
  variantType->addConstructor(getTextSaved(name), functionType);
  return functionType;
}

TypeExpr* Unifier::inferVariantDeclaration(TypeExpr *type, Cursor ast) {
  TRACE();
  auto *variantType = llvm::dyn_cast<VariantOperator>(type);
  ORNULL(variantType);
  auto node = ast.getCurrentNode();
  assert(node.getType() == "variant_declaration");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    if (child.getType() != "constructor_declaration")
      continue;
    auto *ctorType = inferVariantConstructor(variantType, child.getCursor());
    setType(child, ctorType);
  }
  return variantType;
}

TypeExpr* Unifier::inferTypeBinding(Cursor ast) {
  TRACE();
  TypeVarEnvScope scope(typeVarEnv);
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_binding");
  auto namedChildren = getNamedChildren(node);
  auto *namedIterator = namedChildren.begin();
  SmallVector<TypeExpr*> typeVars;
  std::string interfaceStr;
  llvm::raw_string_ostream interface(interfaceStr);
  interface << "type";
  auto savedConcreteTypes = concreteTypes;
  for (auto child = *namedIterator; child.getType() == "type_variable"; child = *++namedIterator) {
    auto *typeVar = declareTypeVariable(getTextSaved(child));
    typeVars.push_back(typeVar);
    interface << " " << *typeVar;
  }
  auto name = node.getChildByFieldName("name");
  interface << " " << getTextSaved(name);
  auto equation = toOptional(node.getChildByFieldName("equation"));
  auto body = toOptional(node.getChildByFieldName("body"));
  auto typeName = getTextSaved(name);
  TypeExpr *thisType = createTypeOperator(typeName, typeVars);
  std::optional<StringRef> eqName = std::nullopt;;
  if (equation) {
    // This is a type alias and we can disregard the original type operator.
    // we MUST disregard the original type operator so type aliases are fully
    // transparent.
    DBGS("Type binding has an equation\n");
    thisType = infer(*equation);
    ORNULL(thisType);
    DBGS("equation: " << *thisType << '\n');
    if (auto *to = llvm::dyn_cast<TypeOperator>(thisType)) {
      eqName = to->getName();
    }
    interface << " = " << *thisType;
  } 
  if (body) {
    DBGS("has body\n");
    if (body->getType() == "variant_declaration") {
      // Variant constructors return the type of the variant, so declare it before inferring
      // the full variant type.
      auto *variantType = create<VariantOperator>(eqName.value_or(typeName), typeVars);
      ORNULL(declareType(name, variantType));
      if (eqName) {
        ORNULL(declareType(eqName.value(), variantType));
      }
      ORNULL(inferVariantDeclaration(variantType, body->getCursor()));
      setType(node, variantType);
      concreteTypes = savedConcreteTypes;
      interface << " = " << variantType->decl();
      saveInterfaceDecl(interface.str());
      return variantType;
    } else if (body->getType() == "record_declaration") {
      // Record declarations need the full record type inferred before declaring
      // the record type.
      auto *recordType = inferRecordDeclaration(typeName, body->getCursor());
      ORNULL(recordType);
      interface << " = " << recordType->decl();
      concreteTypes = savedConcreteTypes;
      saveInterfaceDecl(interface.str());
      return declareType(name, recordType);
    } else {
      RNULL("Unknown type binding body type", *body);
    }
  } else {
    DBGS("No body, type alias to type constructor: " << *thisType << '\n');
    concreteTypes = savedConcreteTypes;
    ORNULL(declareType(name, thisType));
    saveInterfaceDecl(interface.str());
  }
  return thisType;
}

TypeExpr* Unifier::inferConstructedType(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructed_type");
  auto children = getNamedChildren(node);
  auto name = children.pop_back_val();
  auto typeArgs = llvm::map_to_vector(children, [&](Node child) {
    return infer(child);
  });
  auto text = getTextSaved(name);
  if (auto *type = maybeGetDeclaredType(text)) {
    DBGS("Type already exists: " << *type << '\n');
    if (auto *typeOperator = llvm::dyn_cast<TypeOperator>(type)) {
      DBGS("Type operator found\n");
      auto toArgs = typeOperator->getArgs();
      DBG(llvm::for_each(toArgs, [&](auto arg) { DBGS("toArg: " << *arg << '\n'); }));
      DBG(llvm::for_each(typeArgs, [&](auto arg) { DBGS("typeArg: " << *arg << '\n'); }));
      assert(toArgs.size() == typeArgs.size() &&
             "Type operator has different number of arguments than type "
             "arguments");
      for (auto [toArg, typeArg] : llvm::zip(toArgs, typeArgs)) {
        ORNULL(toArg);
        ORNULL(typeArg);
        RNULL_IF(failed(unify(toArg, typeArg)),
                 "Failed to unify type operator argument");
      }
      return typeOperator;
    }
    return type;
  }
  auto *inferredType = createTypeOperator(text, typeArgs);
  return inferredType;
}

TypeExpr* Unifier::inferParenthesizedPattern(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "parenthesized_pattern");
  auto *type = infer(node.getNamedChild(0));
  return type;
}

TypeExpr *Unifier::inferValuePattern(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_pattern");
  auto text = getTextSaved(node);
  if (text == "_") {
    return getWildcardType();
  }
  return declareVariable(node, createTypeVariable());
}

TypeExpr* Unifier::inferExternal(Cursor ast) {
  TypeVarEnvScope scope(typeVarEnv);
  auto node = ast.getCurrentNode();
  auto name = node.getNamedChild(0);
  auto typeNode = node.getNamedChild(1);
  auto type = infer(typeNode);
  return declareVariable(name, type);
}
TypeExpr* Unifier::inferType(Cursor ast) {
  auto node = ast.getCurrentNode();
  static constexpr std::string_view passthroughTypes[] = {
    "parenthesized_expression",
    "then_clause",
    "else_clause",
    "value_definition",
    "expression_item",
    "parenthesized_type",
  };
  if (node.getType() == "number") {
    auto text = getText(node);
    return text.contains('.') ? getDeclaredType("float") : getDeclaredType("int");
  } else if (node.getType() == "float") {
    return getDeclaredType("float");
  } else if (node.getType() == "boolean") {
    return getBoolType();
  } else if (node.getType() == "unit") {
    return getUnitType();
  } else if (node.getType() == "string") {
    return getDeclaredType("string");
  } else if (node.getType() == "guard") {
    return inferGuard(std::move(ast));
  } else if (node.getType() == "do_clause") {
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(std::move(ast));
  } else if (node.getType() == "let_binding") {
    return inferLetBinding(std::move(ast));
  } else if (node.getType() == "let_expression") {
    return inferLetExpression(std::move(ast));
  } else if (node.getType() == "fun_expression") {
    return inferFunctionExpression(std::move(ast));
  } else if (node.getType() == "match_expression") {
    return inferMatchExpression(std::move(ast));
  } else if (node.getType() == "value_path") {
    return inferValuePath(std::move(ast));
  } else if (node.getType() == "for_expression") {
    return inferForExpression(std::move(ast));
  } else if (node.getType() == "infix_expression") {
    return inferInfixExpression(std::move(ast));
  } else if (node.getType() == "if_expression") {
    return inferIfExpression(std::move(ast));
  } else if (node.getType() == "array_expression") {
    return inferArrayExpression(std::move(ast));
  } else if (node.getType() == "array_get_expression") {
    return inferArrayGetExpression(std::move(ast));
  } else if (node.getType() == "list_expression") {
    return inferListExpression(std::move(ast));
  } else if (llvm::is_contained(passthroughTypes, node.getType())) {
    assert(node.getNumNamedChildren() == 1);
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "compilation_unit") {
    return inferCompilationUnit(std::move(ast));
  } else if (node.getType() == "application_expression") {
    return inferApplicationExpression(std::move(ast));
  } else if (node.getType() == "module_definition") {
    return inferModuleDefinition(std::move(ast));
  } else if (node.getType() == "value_specification") {
    return inferValueSpecification(std::move(ast));
  } else if (node.getType() == "type_constructor_path") {
    return inferTypeConstructorPath(std::move(ast));
  } else if (node.getType() == "record_declaration") {
    return inferRecordDeclaration(std::move(ast));
  } else if (node.getType() == "sequence_expression") {
    return inferSequenceExpression(std::move(ast));
  } else if (node.getType() == "type_definition") {
    return inferTypeDefinition(std::move(ast));
  } else if (node.getType() == "value_pattern") {
    return inferValuePattern(std::move(ast));
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(node.getCursor());
  } else if (node.getType() == "parenthesized_pattern") {
    return inferParenthesizedPattern(std::move(ast));
  } else if (node.getType() == "constructor_pattern") {
    return inferConstructorPattern(std::move(ast));
  } else if (node.getType() == "record_pattern") {
    return inferRecordPattern(node.getCursor());
  } else if (node.getType() == "tuple_pattern") {
    return inferTuplePattern(node);
  } else if (node.getType() == "type_variable") {
    return getTypeVariable(node);
  } else if (node.getType() == "constructed_type") {
    return inferConstructedType(std::move(ast));
  } else if (node.getType() == "product_expression") {
    return inferProductExpression(std::move(ast));
  } else if (node.getType() == "function_type") {
    return inferFunctionType(std::move(ast));
  } else if (node.getType() == "open_module") {
    return inferOpenModule(std::move(ast));
  } else if (node.getType() == "external") {
    return inferExternal(std::move(ast));
  } else if (node.getType() == "tuple_type") {
    return inferTupleType(std::move(ast));
  } else if (node.getType() == "tuple_expression") {
    return inferTupleExpression(std::move(ast));
  } else if (node.getType() == "labeled_argument_type") {
    return inferLabeledArgumentType(std::move(ast));
  } else if (node.getType() == "record_expression") {
    return inferRecordExpression(std::move(ast));
  } else if (node.getType() == "field_get_expression") {
    return inferFieldGetExpression(std::move(ast));
  }
  const auto message = SSWRAP("Unknown node type: " << node.getType());
  llvm::errs() << message << '\n';
  assert(false && "Unknown node type");
  RNULL(message, node);
}

} // namespace ocamlc2
