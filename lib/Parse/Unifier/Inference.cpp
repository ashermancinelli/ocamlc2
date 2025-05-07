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

#define DEBUG_TYPE "Inference.cpp"
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
  DBG(show(cursor.copy(), true));
  return te;
}

TypeExpr* Unifier::inferOpenModule(Cursor ast) {
  TRACE();
  auto name = ast.getCurrentNode().getNamedChild(0);
  auto path = getPathParts(name);
  auto *module = getVariableType(path);
  ORNULL(module);
  auto *moduleOperator = llvm::dyn_cast<ModuleOperator>(module);
  RNULL_IF(not moduleOperator, SSWRAP("Expected module, got " << *module));
  moduleStack.back()->openModule(moduleOperator);
  return moduleOperator;
}

TypeExpr* Unifier::inferIncludeModule(Cursor ast) {
  TRACE();
  auto name = ast.getCurrentNode().getNamedChild(0);
  auto path = getPathParts(name);
  auto *module = getVariableType(path);
  ORNULL(module);
  auto *moduleOperator = llvm::dyn_cast<ModuleOperator>(module);
  RNULL_IF(not moduleOperator, SSWRAP("Expected module, got " << *module));
  auto exports = moduleOperator->getExports();
  for (auto exported : exports) {
    DBGS("Export: " << exported.name << '\n');
    if (exported.kind == SignatureOperator::Export::Kind::Variable) {
      exportVariable(exported.name, exported.type);
    } else if (exported.kind == SignatureOperator::Export::Kind::Type) {
      exportType(exported.name, exported.type);
    } else {
      RNULL(SSWRAP("Unknown export kind: " << exported.kind));
    }
  }
  return moduleOperator;
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
    ORNULL(type);
    pattern = pattern.getChildByFieldName("pattern");
    setType(pattern, type);
    return declareVariable(pattern, type);
  } else {
    TRACE();
    TypeExpr *type;
    if (desc.type.has_value()) {
      TRACE();
      type = infer(desc.type.value());
      ORNULL(type);
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
    ORNULL(type);
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
  auto [returnType, types] = [&]() -> std::pair<TypeExpr*, SmallVector<TypeExpr*>> {
    detail::Scope scope(this);
    SmallVector<TypeExpr*> types = llvm::map_to_vector(llvm::zip(parameterDescriptors, parameters), [&](auto arg) -> TypeExpr* {
      auto [desc, param] = arg;
      return declareFunctionParameter(desc, param);
    });
    if (llvm::any_of(types, [](auto *t) { return t == nullptr; })) {
      return std::make_pair(nullptr, types);
    }
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
    RNULL_IF(llvm::any_of(types, [](auto *t) { return t == nullptr; }),
             "Failed to infer function parameter type");
    auto *bodyType = infer(body);
    ORNULL(bodyType);
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
  if (name.getType() == "unit") {
    UNIFY_OR_RNULL(bodyType, getUnitType());
    setType(name, bodyType);
  } else {
    declareVariable(name, bodyType);
  }
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
  const auto currentModule = saveString(filePathToModuleName(sources.back().filepath));
  detail::ModuleScope scope(*this, currentModule);
  if (ast.gotoFirstChild()) {
    do {
      if (!shouldSkip(ast.getCurrentNode())) {
        ORNULL(infer(ast.copy()));
      }
    } while (ast.gotoNextSibling());
  }
  if (!isLoadingStdlib) {
    modulesToDump.push_back(moduleStack.back());
  }
  return moduleStack.back();
}

TypeExpr* Unifier::inferValueDefinition(Cursor ast) {
  TRACE();
  auto children = getNamedChildren(ast.getCurrentNode());
  if (children.size() == 1) {
    return infer(children[0]);
  } else if (children.size() == 2) {
    auto op = children[0];
    auto argument = children[1];
    if (op.getType() == "let_operator") {
      return inferLetOperatorApplication(getTextSaved(op), argument);
    }
  }
  assert(false);
}

TypeExpr* Unifier::inferLetOperatorApplication(llvm::StringRef op, Node argument) {
  assert(false);
  assert(argument.getType() == "let_binding");
  static constexpr llvm::StringRef allowedOps[] = {"let*", "let+"};
  if (!llvm::is_contained(allowedOps, op)) {
    RNULL(SSWRAP("Invalid let operator: " << op));
  }
  auto *opType = getVariableType(op);
  RNULL_IF_NULL(opType, "Invalid let operator");
  auto *bindType = getFunctionType(
      {createTypeVariable(), createTypeVariable(), createTypeVariable()});
  UNIFY_OR_RNULL(opType, bindType);
  // auto pattern = argument.getChildByFieldName("pattern");
  // declareVariable(pattern, bodyType);
  // auto body = argument.getChildByFieldName("body");
  // auto *bodyType = infer(body);
  // RNULL_IF_NULL(bodyType, "Invalid let operator body expression");
  // auto *inferredLetOperatorType = getFunctionType()
  return nullptr;
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

TypeExpr* Unifier::inferModuleTypePath(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_type_path");
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
  setType(op, opType);
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

std::optional<detail::UserDefinedLetOperator> Unifier::isUserDefinedLetOperator(Node node) {
  if (node.getType() == "let_expression") {
    auto children = getNamedChildren(node, {"value_definition"});
    if (!children.empty()) {
      assert(children.size() == 1 && "Expected exactly one value definition in let expression");
      auto letBinding = getNamedChildren(children.front(), {"let_operator", "let_binding"});
      if (letBinding.size() == 2) {
        auto op = letBinding.front();
        assert(op.getType() == "let_operator");
        auto binding = letBinding.back();
        assert(binding.getType() == "let_binding");
        auto exprBody = node.getChildByFieldName("body");
        return detail::UserDefinedLetOperator{
          getTextSaved(op),
          binding.getChildByFieldName("pattern"),
          binding.getChildByFieldName("body"),
          exprBody
        };
      }
    }
  }
  return std::nullopt;
}

TypeExpr* Unifier::inferUserDefinedLetExpression(detail::UserDefinedLetOperator letOperator) {
  TRACE();
  static constexpr llvm::StringRef allowedOps[] = {"let*", "let+"};
  auto [op, bindingPattern, bindingBody, exprBody] = letOperator.tuple();
  assert(llvm::is_contained(allowedOps, op) && "Expected user defined let operator");
  auto *opType = getVariableType(op);
  ORNULL(opType);
  DBGS("Declared let operator type: " << *opType << '\n');

  // 
  // let ( let* ) o f =
  //   match o with
  //     | None -> None
  //     | Some x -> f x
  // let return x = Some x
  // 
  // # val ( let* ) : 'a option -> ('a -> 'b option) -> 'b option = <fun>
  // # val return : 'a -> 'a option = <fun>

  // Match the monadic bind/>>= pattern:
  // val ( let* ) : 'a option -> ('a -> 'b option) -> 'b option
  //                ^ am          ^ a   ^ bm          ^ bm
  //                             ^^^^^^^^^^^^^^^^^ ft

  auto *am = createTypeVariable();
  auto *a = createTypeVariable();
  auto *bm = createTypeVariable();
  auto *ft = getFunctionType({a, bm});
  auto *letOperatorInferredType = getFunctionType({am, ft, bm});
  DBGS("Inferred let operator type: " << *letOperatorInferredType << '\n');

  UNIFY_OR_RNULL(opType, letOperatorInferredType);
  DBGS("Inferred let operator type: " << *letOperatorInferredType << '\n');

  DBGS("Inferred binding body type\n");
  auto *bindingBodyType = infer(bindingBody);
  ORNULL(bindingBodyType);
  DBGS("Inferred binding body type: " << *bindingBodyType << '\n');
  UNIFY_OR_RNULL(bindingBodyType, am);
  DBGS("Inferred binding body type after unification: " << *bindingBodyType << '\n');

  {
    detail::Scope scope(this);
    auto *patternType = declareConcreteVariable(bindingPattern);
    DBGS("Declared pattern type: " << *patternType << '\n');
    auto *inferredExprBodyType = infer(exprBody);
    ORNULL(inferredExprBodyType);
    DBGS("Inferred expr body type: " << *inferredExprBodyType << '\n');
    UNIFY_OR_RNULL(inferredExprBodyType, bm);
    UNIFY_OR_RNULL(patternType, a);
    DBGS("Inferred expr body type after unification: " << *inferredExprBodyType << '\n');
    (void)patternType;
  }

  return bm;
}

TypeExpr* Unifier::inferLetExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "let_expression");
  if (auto udlo = isUserDefinedLetOperator(node)) {
    return inferUserDefinedLetExpression(udlo.value());
  }
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

SignatureOperator* Unifier::inferModuleTypeDefinition(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_type_definition");
  auto name = node.getNamedChild(0);
  assert(!name.isNull() && "Expected module name");
  assert(name.getType() == "module_type_name" && "Expected module name");
  auto body = node.getChildByFieldName("body");
  auto *sig = [&] -> SignatureOperator* {
    detail::ModuleScope ms{*this, getText(name)};
    detail::ConcreteTypeVariableScope concreteScope(*this);
    for (auto child : getNamedChildren(body)) {
      DBGS("Inferring module type definition child: " << child.getType() << '\n');
      ORNULL(infer(child));
    }
    return create<SignatureOperator>(*moduleStack.back());
  }();
  sig->setModuleType();
  declareVariable(name, sig);
  return sig;
}

TypeExpr* Unifier::inferModuleDefinition(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_definition");
  TypeVarEnvScope scope(typeVarEnv);
  auto *moduleBinding = inferModuleBinding(node.getNamedChild(0).getCursor());
  ORNULL(moduleBinding);
  setType(node, moduleBinding);
  return moduleBinding;
}

FailureOr<std::pair<llvm::StringRef, SignatureOperator *>> Unifier::inferModuleParameter(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_parameter");
  auto nameNode = node.getNamedChild(0);
  assert(!nameNode.isNull() && "Expected module parameter name");
  assert(nameNode.getType() == "module_name" && "Expected module parameter name");
  auto name = getTextSaved(nameNode);
  DBGS("Got module parameter name: " << name << '\n');
  auto typeNode = node.getChildByFieldName("module_type");
  assert(!typeNode.isNull() && "Expected module type");
  assert(typeNode.getType() == "module_type_path" && "Expected module type path");
  auto *type = infer(typeNode);
  FAIL_IF(type == nullptr, "Failed to infer module type");
  auto *sigType = llvm::dyn_cast<SignatureOperator>(type);
  FAIL_IF(sigType == nullptr, "Expected module type to be a signature");
  DBGS("Got module parameter type: (" << name << " : " << sigType->getName() << ")\n");
  return success(std::make_pair(name, sigType));
}

TypeExpr* Unifier::inferModuleApplication(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_application");
  auto *module = infer(node.getNamedChild(0));
  ORNULL(module);
  auto *functorExpr = infer(node.getChildByFieldName("functor"));
  ORNULL(functorExpr);
  auto *functor = llvm::dyn_cast<FunctorOperator>(prune(functorExpr));
  RNULL_IF(functor == nullptr, "Expected functor");
  DBGS("Got functor: " << *functor << '\n');

  SmallVector<TypeExpr*> functorApplicationArgs;
  for (unsigned i = 1; i < node.getNumNamedChildren(); ++i) {
    auto *arg = infer(node.getNamedChild(i));
    ORNULL(arg);
    functorApplicationArgs.push_back(arg);
  }

  functorApplicationArgs.push_back(createTypeVariable());
  assert(functorApplicationArgs.size() <= functor->getArgs().size() && "Too many functor arguments");
  auto *appliedFunctor = create<FunctorOperator>("", functorApplicationArgs);

  auto remainingArgs = functor->getArgs().size() - appliedFunctor->getArgs().size();
  if (remainingArgs > 0) {
    DBGS("Normalizing functor from " << *functor << " to match " << *appliedFunctor << '\n');
    auto remainingArgsAndReturnModule = llvm::SmallVector<TypeExpr *>(
        llvm::drop_begin(functor->getArgs(), remainingArgs));
    auto *returnFunctor = create<FunctorOperator>("", remainingArgsAndReturnModule);
    auto args = functor->getArgs();
    auto normalizedArgs = llvm::SmallVector<TypeExpr*>(args.begin(), args.begin() + appliedFunctor->getArgs().size() - 1);
    normalizedArgs.push_back(returnFunctor);
    auto *normalizedFunctor = create<FunctorOperator>("", normalizedArgs);
    functor = normalizedFunctor;
    DBGS("Normalized functor: " << *functor << '\n');
  }

  DBGS("Unifying functor application\n");
  UNIFY_OR_RNULL(appliedFunctor, functor);
  auto *resultingType = appliedFunctor->back();
  DBGS("Got resulting type: " << *resultingType << '\n');
  return resultingType;
}

TypeExpr* Unifier::inferModuleBindingFunctorDefinition(llvm::StringRef name, SmallVector<Node> moduleParameters, SignatureOperator *returnSignature, Node structure) {
  TRACE();
  SmallVector<TypeExpr*> functorTypeArgs;
  SmallVector<std::pair<llvm::StringRef, SignatureOperator*>> functorTypeParams;
  for (auto param : moduleParameters) {
    auto result = inferModuleParameter(param.getCursor());
    RNULL_IF(failed(result), "Failed to infer module parameter");
    auto [paramName, paramType] = result.value();
    // auto *redeclaredSignature = create<SignatureOperator>("", *paramType);
    functorTypeArgs.push_back(paramType);
    functorTypeParams.emplace_back(paramName, paramType);
  }
  auto *functorReturnSignature = inferModuleStructure(structure.getCursor(), functorTypeParams);
  if (returnSignature) {
    UNIFY_OR_RNULL(returnSignature, functorReturnSignature);
  }
  functorTypeArgs.push_back(functorReturnSignature);
  auto *functorType = create<FunctorOperator>(name, functorTypeArgs, functorTypeParams);
  if (returnSignature) {
    functorType->conformsTo(returnSignature);
  }
  declareVariable(name, functorType);
  return functorType;
}

TypeExpr *Unifier::inferModuleBindingModuleDefinition(llvm::StringRef name, SignatureOperator *returnSignature, Node body) {
  TRACE();
  auto nodeType = body.getType();
  DBGS("Body type: " << nodeType << '\n');

  auto *bodyTypeExpr = infer(body);
  ORNULL(bodyTypeExpr);

  auto *resultingSignature = create<ModuleOperator>(name);
  UNIFY_OR_RNULL(bodyTypeExpr, resultingSignature);
  ORNULL(resultingSignature);

  if (nodeType == "module_application") {
    TRACE();
    DBGS("Inferred module result of functor application: " << *resultingSignature << '\n');
    if (returnSignature) {
      UNIFY_OR_RNULL(returnSignature, resultingSignature);
      DBGS("Setting signature conformance: " << *resultingSignature << " to " << *returnSignature << '\n');
      resultingSignature->conformsTo(returnSignature);
    }
    declareVariable(name, resultingSignature);
    return resultingSignature;
  } else if (nodeType == "structure") {
    TRACE();
    if (returnSignature) {
      UNIFY_OR_RNULL(returnSignature, resultingSignature);
      resultingSignature->conformsTo(returnSignature);
    }
    TRACE();

    auto *redeclaredModule = create<ModuleOperator>(name, *resultingSignature);
    DBGS("Redeclared module: " << *redeclaredModule << '\n');
    declareVariable(name, redeclaredModule);
    moduleMap[name] = redeclaredModule;
    return redeclaredModule;
  }
  assert(false && "NYI");
  return nullptr;
}

TypeExpr* Unifier::inferModuleBinding(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  TypeVarEnvScope scope(typeVarEnv);
  detail::ConcreteTypeVariableScope concreteTypeVarScope(*this);
  assert(node.getType() == "module_binding");
  auto name = node.getNamedChild(0);
  assert(!name.isNull() && "Expected module name");
  assert(name.getType() == "module_name" && "Expected module name");
  auto nameText = getTextSaved(name);
  auto signature = toOptional(node.getChildByFieldName("module_type"));
  SignatureOperator *declaredReturnSignature = signature.has_value() ? inferModuleSignature(signature.value().getCursor()) : nullptr;
  auto structure = toOptional(node.getChildByFieldName("body"));
  // TODO: getNamedChildren(node, type="module_parameter") would be nice+save space
  auto moduleParameters = llvm::filter_to_vector(getNamedChildren(node), [](Node n) {
    return n.getType() == "module_parameter";
  });
  if (not moduleParameters.empty()) {
    TRACE();
    assert(structure && "Expected structure on functor");
    return inferModuleBindingFunctorDefinition(nameText, moduleParameters, declaredReturnSignature, *structure);
  } else {
    TRACE();
    return inferModuleBindingModuleDefinition(nameText, declaredReturnSignature, *structure);
  }
}

SignatureOperator* Unifier::inferModuleSignature(Cursor ast) {
  TRACE();
  detail::ModuleScope scope(*this);
  auto node = ast.getCurrentNode();
  auto type = node.getType();
  if (type == "module_type_path") {
    auto *type = infer(node);
    ORNULL(type);
    auto *sigType = llvm::dyn_cast<SignatureOperator>(type);
    RNULL_IF(sigType == nullptr, "Expected module type path to be a signature");
    return sigType;
  }
  assert(type == "signature");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    ORNULL(infer(child));
  }
  return moduleStack.back();
}

ModuleOperator* Unifier::inferModuleStructure(Cursor ast, SmallVector<std::pair<llvm::StringRef, SignatureOperator*>> functorTypeParams) {
  TRACE();
  detail::ModuleScope moduleScope(*this);
  detail::ConcreteTypeVariableScope concreteScope(*this);
  for (auto [paramName, paramType] : functorTypeParams) {
    DBGS("Declaring functor type parameter: " << paramName << " : " << *paramType << '\n');
    localVariable(paramName, paramType); // Declare the module parameter M : S
    
    // Mark abstract types from the parameter signature S as concrete for this scope
    for (const auto& exportItem : paramType->getExports()) {
      if (exportItem.kind == SignatureOperator::Export::Kind::Type) {
        auto* type = prune(exportItem.type);
        if (auto* tv = llvm::dyn_cast<TypeVariable>(type)) {
           DBGS("Marking functor parameter type variable as concrete: " << *tv << " from signature " << paramType->getName() << '\n');
           concreteTypes.insert(tv);
        }
      }
    }
  }
  auto node = ast.getCurrentNode();
  assert(node.getType() == "structure");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    ORNULL(infer(child));
  }
  return moduleStack.back();
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
  return getRecordType(RecordOperator::getAnonRecordName(), {}, fieldTypes, fieldNames);
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
  auto *anonRecordType = getRecordType(RecordOperator::getAnonRecordName(), {}, fieldExpressions, fieldNames);
  ORNULL(anonRecordType);
  auto anonRecordTypes = anonRecordType->getFieldTypes();
  auto anonRecordFieldNames = anonRecordType->getFieldNames();
  for (auto [recordName, seenFieldNames] : seenRecordFields) {
    if (seenFieldNames.size() != fieldExpressions.size()) {
      DBGS("Seen field names size mismatch: " << seenFieldNames.size() << " != " << fieldExpressions.size() << '\n');
      continue;
    }
    auto *maybeSeenRecordType = getDeclaredType(recordName);
    ORNULL(maybeSeenRecordType);
    auto *seenRecordType = llvm::dyn_cast<RecordOperator>(maybeSeenRecordType);
    RNULL_IF_NULL(seenRecordType, "Expected record type");
    auto seenRecordTypes = seenRecordType->getFieldTypes();

    // If we're able to unify the seen record type with what we know about the record type
    // for the current expression, then we have found a match.
    if (llvm::equal(seenFieldNames, anonRecordFieldNames)) {
      if (llvm::all_of_zip(anonRecordTypes, seenRecordTypes, [this](auto *a, auto *b) {
        DBGS("Unifying record field types: " << *a << " and " << *b << '\n');
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

RecordOperator* Unifier::inferRecordDeclaration(llvm::StringRef recordName, SmallVector<TypeExpr*> typeVars, Cursor ast) {
  auto *type = inferRecordDeclaration(std::move(ast));
  ORNULL(type);
  auto *recordType = llvm::dyn_cast<RecordOperator>(type);
  ORNULL(recordType);
  auto fieldNames = recordType->getFieldNames();
  DBGS("Recording field names for record type: " << recordName << '\n');
  seenRecordFields.emplace_back(recordName, fieldNames);
  return getRecordType(recordName, typeVars, recordType->getFieldTypes(), fieldNames);
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
    ORNULL(type);
    auto text = getTextSaved(name);
    if (llvm::find(fieldNames, text) != fieldNames.end()) {
      RNULL("Duplicate field name", name);
    }
    fieldNames.push_back(text);
    fieldTypes.push_back(type);
  }
  return getRecordType(RecordOperator::getAnonRecordName(), {}, fieldTypes, fieldNames);
}

TypeExpr* Unifier::inferTypeConstructorPath(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_constructor_path");
  auto *type = getDeclaredType(node);
  if (isa::uninstantiatedTypeVariable(type)) {
    type = create<TypeAlias>(getTextSaved(node), type);
  }
  ORNULL(type);
  return type;
}

TypeExpr* Unifier::inferTypeDefinition(Cursor ast) {
  TRACE();
  TypeVarEnvScope scope(typeVarEnv);
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_definition");
  TypeExpr *type = nullptr;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    type = inferTypeBinding(child.getCursor());
    ORNULL(type);
    setType(child, type);
  }
  setType(node, type);
  return type;
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
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_binding");
  auto namedChildren = getNamedChildren(node);
  auto *namedIterator = namedChildren.begin();
  SmallVector<TypeExpr*> typeVars;
  for (auto child = *namedIterator; child.getType() == "type_variable"; child = *++namedIterator) {
    auto *typeVar = declareTypeVariable(getTextSaved(child));
    typeVars.push_back(typeVar);
  }
  auto name = node.getChildByFieldName("name");
  auto nameText = getTextSaved(name);
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
    auto *inferredType = infer(*equation);
    ORNULL(inferredType);
    thisType = inferredType;
    DBGS("equation: " << *thisType << '\n');
    if (auto *to = llvm::dyn_cast<TypeOperator>(thisType)) {
      eqName = to->getName();
    }
  } 
  if (body) {
    DBGS("has body\n");
    if (body->getType() == "variant_declaration") {
      TRACE();
      // Variant constructors return the type of the variant, so declare it before inferring
      // the full variant type.
      auto *variantType = create<VariantOperator>(eqName.value_or(typeName), typeVars);
      ORNULL(declareType(name, variantType));
      if (eqName) {
        ORNULL(declareType(eqName.value(), variantType));
      }
      ORNULL(inferVariantDeclaration(variantType, body->getCursor()));
      setType(node, variantType);
      thisType = variantType;
    } else if (body->getType() == "record_declaration") {
      TRACE();
      // Record declarations need the full record type inferred before declaring
      // the record type.
      auto *recordType = inferRecordDeclaration(typeName, typeVars, body->getCursor());
      ORNULL(recordType);
      thisType = declareType(name, recordType);
    } else {
      RNULL("Unknown type binding body type", *body);
    }
  } else {
    if (!equation) {
      DBGS("No equation, creating concrete type variable\n");
      auto *tv = createTypeVariable();
      concreteTypes.insert(tv);
      thisType = tv;
    }
    DBGS("No body, type alias to type constructor: " << nameText << " : " << *thisType << '\n');
    thisType = create<TypeAlias>(nameText, thisType);
    DBGS("Type alias: " << *thisType << " isa operator: " << llvm::isa<TypeOperator>(thisType) << '\n');
    declareType(name, thisType);
  }
  return thisType;
}

TypeExpr* Unifier::inferConstructedType(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructed_type");
  auto children = getNamedChildren(node);
  auto name = children.pop_back_val();
  auto typeArgs = llvm::map_to_vector(children, [&](Node child) -> TypeExpr* {
    DBGS("Inferring type argument for constructed type: " << child.getType() << '\n');
    auto *tv = infer(child);
    return tv;
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
  } else {
    DBGS("Failed to find type operator, creating new one: " << text << '\n');
    auto *inferredType = createTypeOperator(text, typeArgs);
    return inferredType;
  }
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
    "expression_item",
    "parenthesized_type",
    "parenthesized_module_expression",
  };
  auto type = node.getType();
  if (type == "number") {
    auto text = getText(node);
    return text.contains('.') ? getFloatType() : getIntType();
  } else if (type == "float") {
    return getFloatType();
  } else if (type == "boolean") {
    return getBoolType();
  } else if (type == "unit") {
    return getUnitType();
  } else if (type == "string") {
    return getStringType();
  } else if (type == "guard") {
    return inferGuard(std::move(ast));
  } else if (type == "sign_expression") {
    return infer(node.getChildByFieldName("expression"));
  } else if (type == "do_clause") {
    return infer(node.getNamedChild(0));
  } else if (type == "constructor_path") {
    return inferConstructorPath(std::move(ast));
  } else if (type == "module_type_path") {
    return inferModuleTypePath(std::move(ast));
  } else if (type == "let_binding") {
    return inferLetBinding(std::move(ast));
  } else if (type == "let_expression") {
    return inferLetExpression(std::move(ast));
  } else if (type == "fun_expression") {
    return inferFunctionExpression(std::move(ast));
  } else if (type == "match_expression") {
    return inferMatchExpression(std::move(ast));
  } else if (type == "value_path" or type == "module_path") {
    return inferValuePath(std::move(ast));
  } else if (type == "for_expression") {
    return inferForExpression(std::move(ast));
  } else if (type == "infix_expression") {
    return inferInfixExpression(std::move(ast));
  } else if (type == "if_expression") {
    return inferIfExpression(std::move(ast));
  } else if (type == "array_expression") {
    return inferArrayExpression(std::move(ast));
  } else if (type == "array_get_expression") {
    return inferArrayGetExpression(std::move(ast));
  } else if (type == "list_expression") {
    return inferListExpression(std::move(ast));
  } else if (llvm::is_contained(passthroughTypes, type)) {
    assert(node.getNumNamedChildren() == 1);
    return infer(node.getNamedChild(0));
  } else if (type == "compilation_unit") {
    return inferCompilationUnit(std::move(ast));
  } else if (type == "application_expression") {
    return inferApplicationExpression(std::move(ast));
  } else if (type == "module_type_definition") {
    return inferModuleTypeDefinition(std::move(ast));
  } else if (type == "module_definition") {
    return inferModuleDefinition(std::move(ast));
  } else if (type == "value_specification") {
    return inferValueSpecification(std::move(ast));
  } else if (type == "type_constructor_path") {
    return inferTypeConstructorPath(std::move(ast));
  } else if (type == "record_declaration") {
    return inferRecordDeclaration(std::move(ast));
  } else if (type == "sequence_expression") {
    return inferSequenceExpression(std::move(ast));
  } else if (type == "type_definition") {
    return inferTypeDefinition(std::move(ast));
  } else if (type == "value_pattern") {
    return inferValuePattern(std::move(ast));
  } else if (type == "constructor_path") {
    return inferConstructorPath(node.getCursor());
  } else if (type == "parenthesized_pattern") {
    return inferParenthesizedPattern(std::move(ast));
  } else if (type == "constructor_pattern") {
    return inferConstructorPattern(std::move(ast));
  } else if (type == "record_pattern") {
    return inferRecordPattern(node.getCursor());
  } else if (type == "tuple_pattern") {
    return inferTuplePattern(node);
  } else if (type == "type_variable") {
    return getTypeVariable(node);
  } else if (type == "constructed_type") {
    return inferConstructedType(std::move(ast));
  } else if (type == "function_type") {
    return inferFunctionType(std::move(ast));
  } else if (type == "open_module") {
    return inferOpenModule(std::move(ast));
  } else if (type == "include_module") {
    return inferIncludeModule(std::move(ast));
  } else if (type == "module_name") {
    return getVariableType(getTextSaved(node));
  } else if (type == "external") {
    return inferExternal(std::move(ast));
  } else if (type == "tuple_type") {
    return inferTupleType(std::move(ast));
  } else if (type == "tuple_expression") {
    return inferTupleExpression(std::move(ast));
  } else if (type == "labeled_argument_type") {
    return inferLabeledArgumentType(std::move(ast));
  } else if (type == "record_expression") {
    return inferRecordExpression(std::move(ast));
  } else if (type == "field_get_expression") {
    return inferFieldGetExpression(std::move(ast));
  } else if (type == "structure") {
    return inferModuleStructure(std::move(ast));
  } else if (type == "module_application") {
    return inferModuleApplication(std::move(ast));
  } else if (type == "value_definition") {
    return inferValueDefinition(std::move(ast));
  }
  const auto message = SSWRAP("Unknown node type: " << node.getType());
  llvm::errs() << message << '\n';
  assert(false && "Unknown node type");
  RNULL(message, node);
}

} // namespace ocamlc2
