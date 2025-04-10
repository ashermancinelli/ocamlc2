#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <sstream>

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;

// Type implementation methods

std::shared_ptr<Type> TypeVar::substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const {
  auto it = subst.find(name);
  if (it != subst.end()) {
    return it->second;
  }
  return std::make_shared<TypeVar>(name);
}

std::shared_ptr<Type> TypeCon::substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const {
  return std::make_shared<TypeCon>(name);
}

std::string TypeApp::toString() const {
  return constructor->toString() + " " + argument->toString();
}

std::set<std::string> TypeApp::getFreeTypeVars() const {
  auto conVars = constructor->getFreeTypeVars();
  auto argVars = argument->getFreeTypeVars();
  conVars.insert(argVars.begin(), argVars.end());
  return conVars;
}

std::shared_ptr<Type> TypeApp::substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const {
  return std::make_shared<TypeApp>(
    constructor->substitute(subst),
    argument->substitute(subst)
  );
}

std::string TypeArrow::toString() const {
  std::string fromStr = from->toString();
  std::string toStr = to->toString();
  
  // Add parentheses around complex types in 'from' position
  if (from->getKind() == Kind_Arrow) {
    fromStr = "(" + fromStr + ")";
  }
  
  return fromStr + " -> " + toStr;
}

std::set<std::string> TypeArrow::getFreeTypeVars() const {
  auto fromVars = from->getFreeTypeVars();
  auto toVars = to->getFreeTypeVars();
  fromVars.insert(toVars.begin(), toVars.end());
  return fromVars;
}

std::shared_ptr<Type> TypeArrow::substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const {
  return std::make_shared<TypeArrow>(
    from->substitute(subst),
    to->substitute(subst)
  );
}

// TypeScheme implementation
std::string TypeScheme::toString() const {
  std::stringstream ss;
  if (!vars.empty()) {
    ss << "forall ";
    for (size_t i = 0; i < vars.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << vars[i];
    }
    ss << ". ";
  }
  ss << type->toString();
  return ss.str();
}

std::set<std::string> TypeScheme::getFreeTypeVars() const {
  auto freeVars = type->getFreeTypeVars();
  for (const auto& var : vars) {
    freeVars.erase(var);
  }
  return freeVars;
}

std::shared_ptr<Type> TypeScheme::instantiate(TypeInferenceContext& context) const {
  std::unordered_map<std::string, std::shared_ptr<Type>> subst;
  for (const auto& var : vars) {
    subst[var] = context.freshTypeVar();
  }
  return type->substitute(subst);
}

// TypeEnv implementation
std::set<std::string> TypeEnv::getFreeTypeVars() const {
  std::set<std::string> freeVars;
  for (const auto& [_, scheme] : env) {
    auto schemeVars = scheme->getFreeTypeVars();
    freeVars.insert(schemeVars.begin(), schemeVars.end());
  }
  return freeVars;
}

TypeEnv TypeEnv::substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const {
  TypeEnv newEnv;
  for (const auto& [name, scheme] : env) {
    auto type = scheme->getType()->substitute(subst);
    auto vars = scheme->getVars();
    newEnv.extend(name, std::make_shared<TypeScheme>(vars, type));
  }
  return newEnv;
}

// TypeConstraint implementation
std::string TypeConstraint::toString() const {
  return lhs->toString() + " = " + rhs->toString();
}

// TypeInferenceContext implementation
bool TypeInferenceContext::occurs(const std::string& var, const std::shared_ptr<Type>& type) {
  auto freeVars = type->getFreeTypeVars();
  return freeVars.find(var) != freeVars.end();
}

std::unordered_map<std::string, std::shared_ptr<Type>> TypeInferenceContext::unifyOne(
    const std::shared_ptr<Type>& t1, 
    const std::shared_ptr<Type>& t2,
    std::unordered_map<std::string, std::shared_ptr<Type>> subst) {
  
  // Apply current substitution to types
  auto s1 = t1->substitute(subst);
  auto s2 = t2->substitute(subst);
  
  // Case 1: Types are identical
  if (s1->toString() == s2->toString()) {
    return subst;
  }
  
  // Case 2: First type is a variable
  if (llvm::isa<TypeVar>(s1.get())) {
    const auto* var1 = llvm::cast<TypeVar>(s1.get());
    std::string varName = var1->getName();
    
    // Occurs check - prevent infinite types
    if (occurs(varName, s2)) {
      // Instead of throwing, return an error result or log and continue
      llvm::errs() << "Occurs check failed: " << varName << " occurs in " << s2->toString() << "\n";
      return subst; // Return current substitution without adding this binding
    }
    
    // Add substitution
    subst[varName] = s2;
    return subst;
  }
  
  // Case 3: Second type is a variable
  if (llvm::isa<TypeVar>(s2.get())) {
    const auto* var2 = llvm::cast<TypeVar>(s2.get());
    std::string varName = var2->getName();
    
    // Occurs check
    if (occurs(varName, s1)) {
      // Instead of throwing, return an error result or log and continue
      llvm::errs() << "Occurs check failed: " << varName << " occurs in " << s1->toString() << "\n";
      return subst; // Return current substitution without adding this binding
    }
    
    // Add substitution
    subst[varName] = s1;
    return subst;
  }
  
  // Case 4: Both are function types
  if (llvm::isa<TypeArrow>(s1.get()) && llvm::isa<TypeArrow>(s2.get())) {
    const auto* arrow1 = llvm::cast<TypeArrow>(s1.get());
    const auto* arrow2 = llvm::cast<TypeArrow>(s2.get());
    
    // Unify domains
    subst = unifyOne(arrow1->getFrom(), arrow2->getFrom(), subst);
    // Unify codomains
    return unifyOne(arrow1->getTo(), arrow2->getTo(), subst);
  }
  
  // Case 5: Both are type applications
  if (llvm::isa<TypeApp>(s1.get()) && llvm::isa<TypeApp>(s2.get())) {
    const auto* app1 = llvm::cast<TypeApp>(s1.get());
    const auto* app2 = llvm::cast<TypeApp>(s2.get());
    
    // Unify constructors
    subst = unifyOne(app1->getConstructor(), app2->getConstructor(), subst);
    // Unify arguments
    return unifyOne(app1->getArgument(), app2->getArgument(), subst);
  }
  
  // Cannot unify - instead of throwing, log error and return current subst
  llvm::errs() << "Cannot unify types: " << s1->toString() << " and " << s2->toString() << "\n";
  return subst;
}

std::unordered_map<std::string, std::shared_ptr<Type>> TypeInferenceContext::unify(
    const std::vector<TypeConstraint>& constraints) {
  
  std::unordered_map<std::string, std::shared_ptr<Type>> subst;
  
  for (const auto& constraint : constraints) {
    subst = unifyOne(constraint.getLHS(), constraint.getRHS(), subst);
  }
  
  return subst;
}

std::shared_ptr<TypeScheme> TypeInferenceContext::generalize(
    const TypeEnv& env, const std::shared_ptr<Type>& type) {
  
  // Find free type variables in type that are not in environment
  auto typeFreeVars = type->getFreeTypeVars();
  auto envFreeVars = env.getFreeTypeVars();
  
  std::vector<std::string> schemeVars;
  for (const auto& var : typeFreeVars) {
    if (envFreeVars.find(var) == envFreeVars.end()) {
      schemeVars.push_back(var);
    }
  }
  
  return std::make_shared<TypeScheme>(schemeVars, type);
}

std::shared_ptr<Type> TypeInferenceContext::inferType(const ASTNode* node, TypeEnv& env) {
  constraints.clear();
  auto type = inferExpr(node, env);
  
  // No try-catch, just unify and handle errors as they occur
  auto subst = unify(constraints);
  
  // Apply substitution to inferred type
  return type->substitute(subst);
}

std::shared_ptr<Type> TypeInferenceContext::inferExpr(const ASTNode* node, TypeEnv& env) {
  if (!node) {
    return std::make_shared<TypeCon>("unit");
  }
  
  switch (node->getKind()) {
    case ASTNode::Node_Number:
      return inferNumberExpr(static_cast<const NumberExprAST*>(node), env);
    
    case ASTNode::Node_ValuePath:
      return inferValuePath(static_cast<const ValuePathAST*>(node), env);
    
    case ASTNode::Node_Application:
      return inferApplication(static_cast<const ApplicationExprAST*>(node), env);
    
    case ASTNode::Node_InfixExpression:
      return inferInfixExpr(static_cast<const InfixExpressionAST*>(node), env);
    
    case ASTNode::Node_MatchExpression:
      return inferMatchExpr(static_cast<const MatchExpressionAST*>(node), env);
    
    case ASTNode::Node_LetExpression:
      return inferLetExpr(static_cast<const LetExpressionAST*>(node), env);
    
    case ASTNode::Node_ParenthesizedExpression: {
      auto parenExpr = static_cast<const ParenthesizedExpressionAST*>(node);
      return inferExpr(parenExpr->getExpression(), env);
    }
    
    case ASTNode::Node_ExpressionItem: {
      DBGS("ExpressionItemAST: " << node << "\n");
      auto exprItem = static_cast<const ExpressionItemAST*>(node);
      return inferExpr(exprItem->getExpression(), env);
    }
    
    case ASTNode::Node_ValueDefinition: {
      DBGS("ValueDefinitionAST: " << node << "\n");
      auto valueDef = static_cast<const ValueDefinitionAST*>(node);
      std::shared_ptr<Type> result;
      
      for (const auto& binding : valueDef->getBindings()) {
        auto bindingType = inferLetBinding(binding.get(), env);
        
        // Create a generalized scheme for the binding
        auto scheme = generalize(env, bindingType);
        
        // Extend environment with binding
        env.extend(binding->getName(), scheme);
        
        // Return type of the last binding
        result = bindingType;
      }
      
      return result ? result : std::make_shared<TypeCon>("unit");
    }
    
    case ASTNode::Node_ConstructorPath: {
      DBGS("ConstructorPathAST: " << node << "\n");
      auto constructor = static_cast<const ConstructorPathAST*>(node);
      const auto& path = constructor->getPath();
      
      // Special case for boolean literals
      if (path.size() == 1 && (path[0] == "true" || path[0] == "false")) {
        return std::make_shared<TypeCon>("bool");
      }
      
      // For simplicity, treat constructors as fresh type variables
      // In a real implementation, we'd look up the constructor's type
      return freshTypeVar();
    }
    
    case ASTNode::Node_ValuePattern: {
      // For pattern matching, we'd assign a fresh type variable
      return freshTypeVar();
    }
    
    case ASTNode::Node_TypeDefinition: {
      // Type definitions don't produce a value, just return unit
      return std::make_shared<TypeCon>("unit");
    }
    
    case ASTNode::Node_CompilationUnit: {
      auto unit = static_cast<const CompilationUnitAST*>(node);
      std::shared_ptr<Type> result = std::make_shared<TypeCon>("unit");
      
      for (const auto& item : unit->getItems()) {
        result = inferExpr(item.get(), env);
      }
      
      return result;
    }
    
    default:
      // Check for special cases in the node representation
      std::string nodeKindName = ASTNode::getName(*node).str();
      
      // Handle function expressions (lambda)
      if (nodeKindName == "ValuePattern" && 
          static_cast<const ValuePatternAST*>(node)->getName() == "fun") {
        // Create parameter and body types
        auto paramType = freshTypeVar();
        auto bodyType = freshTypeVar();
        
        // Return function type
        return std::make_shared<TypeArrow>(paramType, bodyType);
      }
      
      // Handle boolean literals
      if (nodeKindName == "ValuePattern") {
        const auto& name = static_cast<const ValuePatternAST*>(node)->getName();
        if (name == "true" || name == "false") {
          return std::make_shared<TypeCon>("bool");
        }
      }
      
      llvm::errs() << "Unsupported node kind for type inference: " << nodeKindName << "\n";
      return std::make_shared<TypeCon>("unknown");
  }
}

std::shared_ptr<Type> TypeInferenceContext::inferNumberExpr(
    const NumberExprAST* node, TypeEnv& env) {
  DBGS("NumberExprAST: " << node << "\n");
  // For simplicity, assuming all numbers are integers
  return std::make_shared<TypeCon>("int");
}

std::shared_ptr<Type> TypeInferenceContext::inferValuePath(
    const ValuePathAST* node, TypeEnv& env) {
  DBGS("ValuePathAST: " << node << "\n");
  // For a simple value path, look up in environment
  if (node->getPath().size() == 1) {
    const auto& name = node->getPath()[0];
    auto scheme = env.lookup(name);
    if (scheme) {
      return scheme->instantiate(*this);
    }
    
    // Not found in environment, create a fresh type variable
    // (for demonstration - in a real typechecker, this would be an error)
    auto newVar = freshTypeVar();
    DBGS("Warning: Unbound variable '" << name << "', using fresh type variable "
         << newVar->toString() << "\n");
    return newVar;
  }
  
  // For module-qualified paths, we'd need module handling
  llvm::errs() << "Warning: Module-qualified paths not yet supported in type inference\n";
  return freshTypeVar();
}

std::shared_ptr<Type> TypeInferenceContext::inferApplication(
    const ApplicationExprAST* node, TypeEnv& env) {
  DBGS("ApplicationExprAST: " << node << "\n");
  // Infer type of function
  auto funcType = inferExpr(node->getFunction(), env);
  
  // Infer types of arguments
  std::vector<std::shared_ptr<Type>> argTypes;
  for (const auto& arg : node->getArguments()) {
    argTypes.push_back(inferExpr(arg.get(), env));
  }
  
  // Create result type variable
  auto resultType = freshTypeVar();
  
  // Build function type and add constraint
  std::shared_ptr<Type> expectedFuncType = resultType;
  for (auto it = argTypes.rbegin(); it != argTypes.rend(); ++it) {
    expectedFuncType = std::make_shared<TypeArrow>(*it, expectedFuncType);
  }
  
  constraints.push_back(TypeConstraint(funcType, expectedFuncType));
  
  return resultType;
}

std::shared_ptr<Type> TypeInferenceContext::inferInfixExpr(
    const InfixExpressionAST* node, TypeEnv& env) {
  DBGS("InfixExpressionAST: " << node << "\n");
  auto leftType = inferExpr(node->getLHS(), env);
  auto rightType = inferExpr(node->getRHS(), env);
  const auto& op = node->getOperator();
  
  if (op == "+" || op == "-" || op == "*" || op == "/") {
    // Arithmetic operators
    auto intType = std::make_shared<TypeCon>("int");
    constraints.push_back(TypeConstraint(leftType, intType));
    constraints.push_back(TypeConstraint(rightType, intType));
    return intType;
  } else if (op == "=" || op == "<>" || op == "<" || op == ">" || op == "<=" || op == ">=") {
    // Comparison operators
    constraints.push_back(TypeConstraint(leftType, rightType));
    return std::make_shared<TypeCon>("bool");
  } else if (op == "&&" || op == "||") {
    // Boolean operators
    auto boolType = std::make_shared<TypeCon>("bool");
    constraints.push_back(TypeConstraint(leftType, boolType));
    constraints.push_back(TypeConstraint(rightType, boolType));
    return boolType;
  } else if (op == "::") {
    // List cons operator
    auto listType = std::make_shared<TypeApp>(
      std::make_shared<TypeCon>("list"),
      leftType
    );
    constraints.push_back(TypeConstraint(rightType, listType));
    return listType;
  }
  
  // Unknown operator, create a general constraint
  auto resultType = freshTypeVar();
  auto opFuncType = std::make_shared<TypeArrow>(
    leftType,
    std::make_shared<TypeArrow>(
      rightType,
      resultType
    )
  );
  
  // If we had an environment with built-in operators, we would look it up here
  
  return resultType;
}

std::shared_ptr<Type> TypeInferenceContext::inferMatchExpr(
    const MatchExpressionAST* node, TypeEnv& env) {
  DBGS("MatchExpressionAST: " << node << "\n");
  // Infer type of value being matched
  auto valueType = inferExpr(node->getValue(), env);
  
  // Create a fresh type variable for the result
  auto resultType = freshTypeVar();
  
  // Process each case
  for (const auto& matchCase : node->getCases()) {
    // Pattern matching would need more extensive handling
    // For now, we'll assume patterns don't introduce bindings
    
    // Infer type of case expression
    auto caseType = inferExpr(matchCase->getExpression(), env);
    
    // All case expressions should have the same type
    constraints.push_back(TypeConstraint(resultType, caseType));
  }
  
  return resultType;
}

std::shared_ptr<Type> TypeInferenceContext::inferLetExpr(
    const LetExpressionAST* node, TypeEnv& env) {
  DBGS("LetExpressionAST: " << node << "\n");
  // A let expression has the form "let binding in body"
  auto binding = node->getBinding();
  auto body = node->getBody();
  
  // Handle let-binding
  if (auto letBinding = llvm::dyn_cast<LetBindingAST>(binding)) {
    // Infer type of bound value
    auto bindingType = inferLetBinding(letBinding, env);
    
    // Get name of binding
    const auto& bindingName = letBinding->getName();
    
    // Create a generalized scheme for the binding
    auto scheme = generalize(env, bindingType);
    
    // Extend environment with binding for body
    TypeEnv bodyEnv = env;
    bodyEnv.extend(bindingName, scheme);
    
    // Infer type of body in extended environment
    return inferExpr(body, bodyEnv);
  } else {
    // Not a proper let binding, just infer body
    return inferExpr(body, env);
  }
}

std::shared_ptr<Type> TypeInferenceContext::inferLetBinding(
    const LetBindingAST* node, TypeEnv& env) {
  DBGS("LetBindingAST: " << node << "\n");
  // Don't store the name if we're not using it
  auto body = node->getBody();
  const auto& params = node->getParameters();
  
  // Create a fresh type variable for the result
  auto resultType = freshTypeVar();
  
  // Create new environment for function body
  TypeEnv funcEnv = env;
  
  // Handle parameters
  std::vector<std::shared_ptr<Type>> paramTypes;
  for (const auto& param : params) {
    auto paramType = freshTypeVar();
    paramTypes.push_back(paramType);
    
    // If parameter has a name, add to environment
    if (auto valuePattern = llvm::dyn_cast<ValuePatternAST>(param.get())) {
      const auto& paramName = valuePattern->getName();
      if (paramName != "_") {  // Skip wildcard
        auto scheme = std::make_shared<TypeScheme>(
          std::vector<std::string>{}, paramType);
        funcEnv.extend(paramName, scheme);
      }
    }
    // Handle typed patterns if needed
  }
  
  // Infer type of function body
  auto bodyType = inferExpr(body, funcEnv);
  
  // Build function type
  std::shared_ptr<Type> funcType = bodyType;
  for (auto it = paramTypes.rbegin(); it != paramTypes.rend(); ++it) {
    funcType = std::make_shared<TypeArrow>(*it, funcType);
  }
  
  // If there's a return type annotation, add constraint
  if (auto returnType = node->getReturnType()) {
    // Convert returnType to a Type (simplified)
    auto typeStr = returnType->getPath()[0];
    auto explicitType = std::make_shared<TypeCon>(typeStr);
    constraints.push_back(TypeConstraint(bodyType, explicitType));
  }
  
  return funcType;
}

// Utility functions
std::shared_ptr<TypeScheme> ocamlc2::inferProgramType(const ASTNode* ast) {
  TypeInferenceContext context;
  TypeEnv env;
  
  // Add built-in types and functions to environment
  
  // Arithmetic operators
  env.extend("+", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("int"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("int"),
        std::make_shared<TypeCon>("int")
      )
    )
  ));
  
  env.extend("-", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("int"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("int"),
        std::make_shared<TypeCon>("int")
      )
    )
  ));
  
  env.extend("*", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("int"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("int"),
        std::make_shared<TypeCon>("int")
      )
    )
  ));
  
  env.extend("/", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("int"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("int"),
        std::make_shared<TypeCon>("int")
      )
    )
  ));
  
  // Comparison operators
  auto alpha = std::make_shared<TypeVar>("'a");
  env.extend("=", std::make_shared<TypeScheme>(
    std::vector<std::string>{"'a"},
    std::make_shared<TypeArrow>(
      alpha,
      std::make_shared<TypeArrow>(
        alpha,
        std::make_shared<TypeCon>("bool")
      )
    )
  ));
  
  // Boolean operators
  env.extend("&&", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("bool"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("bool"),
        std::make_shared<TypeCon>("bool")
      )
    )
  ));
  
  env.extend("||", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("bool"),
      std::make_shared<TypeArrow>(
        std::make_shared<TypeCon>("bool"),
        std::make_shared<TypeCon>("bool")
      )
    )
  ));
  
  // Standard library functions
  env.extend("print_int", std::make_shared<TypeScheme>(
    std::vector<std::string>{},
    std::make_shared<TypeArrow>(
      std::make_shared<TypeCon>("int"),
      std::make_shared<TypeCon>("unit")
    )
  ));
  
  // Infer type
  auto type = context.inferType(ast, env);
  
  // Generalize to create a type scheme
  return context.generalize(env, type);
}

void ocamlc2::dumpType(const std::shared_ptr<Type>& type) {
  llvm::outs() << type->toString() << "\n";
}

void ocamlc2::dumpTypeScheme(const std::shared_ptr<TypeScheme>& scheme) {
  llvm::outs() << scheme->toString() << "\n";
} 
