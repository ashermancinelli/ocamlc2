#include "ocamlc2/Parse/MLIRGen.h"
#include <mlir/Support/LLVM.h>
#include <tree_sitter/api.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Attributes.h>
#include "ocamlc2/Parse/TSAdaptor.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <string_view>
#include <ocamlc2/Parse/Runtime.h>
#include "ocamlc2/Dialect/OcamlOpBuilder.h"

#define DEBUG_TYPE "mlirgen"
#include "ocamlc2/Support/Debug.h.inc"

FailureOr<std::vector<mlir::Type>> MLIRGen::getPrintfTypeHints(mlir::ValueRange args, TSNode *stringContentNode) {
  auto children = childrenNodes(*stringContentNode);
  std::vector<mlir::Type> typeHints;
  for (auto [childType, child] : children) {
    if (childType == "conversion_specification") {
      auto text = adaptor->text(&child);
      if (text == "%d") {
        typeHints.push_back(builder.getI32Type());
      } else if (text == "%f") {
        typeHints.push_back(builder.getF64Type());
      } else if (text == "%s") {
        typeHints.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));
      } else {
        return emitError(loc(*stringContentNode)) << "Unhandled conversion specification: " << text;
      }
    }
  }
  return typeHints;
}

MLIRGen::MLIRGen(mlir::MLIRContext &context)
    : context(context), builder(&context) {
}

mlir::Location MLIRGen::loc(TSNode node) {
  auto pt = ts_node_start_point(node);
  return mlir::FileLineColLoc::get(builder.getStringAttr(adaptor->getFilename()), pt.row, pt.column);
}

LogicalResult MLIRGen::declareValue(llvm::StringRef name, mlir::Value value) {
  DBGS("declaring " << name << " as " << value << "\n");
  symbolTable.insert(name, value);
  return mlir::success();
}

FailureOr<mlir::Value> MLIRGen::lookupValuePath(TSNode *node) {
  if (auto ident = valuePathToIdentifier(node); succeeded(ident)) {
    return lookupValue(ident.value());
  }
  return failure();
}

FailureOr<mlir::Value> MLIRGen::lookupValue(llvm::StringRef name, std::optional<mlir::Location> maybeLoc) {
  if (auto var = symbolTable.lookup(name)) {
    return var;
  }
  llvm::errs() << "Variable not found: " << name << "\n";
  mlir::Location loc = maybeLoc.value_or(mlir::UnknownLoc::get(&getContext()));
  return mlir::emitError(loc) << "Variable not found: " << name;
}

FailureOr<mlir::Value> MLIRGen::genRuntimeCall(llvm::StringRef name, mlir::ValueRange args, mlir::Location loc, TSNode *node) {
  auto runtimeFunctions = RuntimeFunction::getRuntimeFunctions();
  auto it = llvm::find_if(runtimeFunctions, [&](const RuntimeFunction &func) {
    return mangleIdentifier(func.name) == name;
  });
  if (it == runtimeFunctions.end()) {
    return failure();
  }
  return it->call(this, node, loc, args);
}

std::string MLIRGen::mangleIdentifier(llvm::StringRef name) {
  return std::string(name);
}


FailureOr<std::string> MLIRGen::valuePathToIdentifier(TSNode *node) {
  std::string text = adaptor->text(node);
  return mangleIdentifier(text);
}

std::string MLIRGen::sanitizeParsedString(TSNode *node) {
  auto children = childrenNodes(*node);
  assert(children.size() == 3);
  auto stringContentNode = children[1].second;
  auto stringComponents = childrenNodes(stringContentNode);
  std::stringstream ss;
  for (auto [childType, child] : stringComponents) {
    if (childType == "conversion_specification") {
      ss << adaptor->text(&child);
    } else if (childType == "escape_sequence") {
      auto escapeSequence = adaptor->text(&child);
      if (escapeSequence == "\\n") {
        ss << '\n';
      } else if (escapeSequence == "\\t") {
        ss << '\t';
      } else if (escapeSequence == "\\\\") {
        ss << '\\';
      } else {
        llvm::errs() << "Unhandled escape sequence: " << escapeSequence << "\n";
      }
    } else {
      llvm::errs() << "Unhandled string component: " << childType << "\n";
    }
  }
  return ss.str();
}

std::string MLIRGen::getUniqueName(std::string_view prefix) {
  static long long unsigned counter = 0;
  return llvm::formatv("{0}ID{1:04}",
                       (prefix.size() ? std::string(prefix) + "_" : ""),
                       counter++)
      .str();
}

FailureOr<std::vector<Argument>> MLIRGen::getFunctionArguments(NodeIter it) {
  TRACE();
  std::vector<Argument> arguments;
  assert(it->first == "parameter");
  while (it->first == "parameter") {
    auto parameterChildren = childrenNodes(it++->second);
    Argument argument;
    if (parameterChildren[0].first == "value_pattern") {
      auto node = parameterChildren[0].second;
      auto text = adaptor->text(&node);
      DBGS("value pattern: " << text << "\n");
      argument.first = text;
    } else if (parameterChildren[0].first == "typed_pattern") {
      auto node = parameterChildren[0].second;
      auto text = adaptor->text(&node);
      auto children = childrenNodes(node);
      auto valuePattern = children[1].second;
      auto typeCtorPath = children[3].second;
      argument.first = adaptor->text(&valuePattern);
      argument.second = valuePathToIdentifier(&typeCtorPath);
      DBGS("typed pattern: " << argument.first << " " << argument.second << "\n");
    } else {
      return emitError(loc(parameterChildren[0].second)) << "Unhandled parameter pattern";
    }
    arguments.push_back(argument);
  }
  DBGS("got " << arguments.size() << " function arguments\n");
  return arguments;
}

FailureOr<mlir::func::FuncOp> MLIRGen::lookupFunction(llvm::StringRef name) {
  if (auto func = module->lookupSymbol<mlir::func::FuncOp>(name)) {
    return func;
  }
  return failure();
}

FailureOr<mlir::Value> MLIRGen::gen(NodeIter it) {
  auto [childType, child] = *it;
  auto text = adaptor->text(&child);
  DBGS(childType << " " << (text.contains("\n") ? "" : text) << "\n");
  if (childType == "comment" or childType == ";;") {
    return mlir::Value();
  } else if (childType == "value_path") {
    return lookupValuePath(&child);
  } else if (childType == "typed_expression") {
    auto children = childrenNodes(child);
    auto childIter = children.begin();
    assert(childIter++->first == "(");
    auto constructorArgument = must(gen(childIter++));
    assert(childIter++->first == ":");
    auto typeConstructorIdentifier = must(valuePathToIdentifier(&childIter++->second));
    assert(childIter++->first == ")");
    assert(childIter == children.end());
    if (auto runtimeCall = genRuntimeCall(typeConstructorIdentifier, {constructorArgument}, loc(child), &child);
        succeeded(runtimeCall)) {
      return runtimeCall;
    } else {
      return emitError(loc(child)) << "type constructor failed";
    }
  } else if (childType == "string") {
    auto sanitizedText = sanitizeParsedString(&child);
    auto textAttr = builder.getStringAttr(sanitizedText);
    auto name = getUniqueName();
    mlir::LLVM::GlobalOp globalOp;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module->getBody());
      auto arrayType = mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), sanitizedText.size());
      auto linkage = mlir::LLVM::Linkage::Internal;
      globalOp = builder.create<mlir::LLVM::GlobalOp>(
          loc(child), arrayType, /*isConstant=*/true, linkage, name, textAttr);
    }
    auto addrOfOp = builder.create<mlir::LLVM::AddressOfOp>(loc(child), globalOp);
    auto intrinsicName = builder.getStringAttr("embox_string");
    auto emboxed = builder.create<mlir::ocaml::IntrinsicOp>(
        loc(child), mlir::ocaml::StringType::get(builder.getContext()), intrinsicName,
        mlir::ValueRange{addrOfOp.getResult()});
    return emboxed.getResult();
  } else if (childType == "let_binding") {
    auto letChildren = childrenNodes(child);
    if (letChildren.size() == 3) {
      DBGS("value declaration\n");
      auto rhs = must(gen(letChildren.begin() + 2));
      if (letChildren[0].first == "unit") {
        return rhs;
      }
      return must(genAssign(letChildren[0].first, rhs));
    } else {
      DBGS("function definition\n");
      auto it = letChildren.begin();
      assert(it->first == "value_name");
      auto callee = adaptor->text(&it++->second);
      auto arguments = must(getFunctionArguments(it));
      it += arguments.size();
      assert(it++->first == "=");
      auto oboxType = builder.getOBoxType();
      SmallVector<mlir::Type> argumentTypes;
      for (auto [identifier, maybeTypeCtor] : arguments) {
        if (maybeTypeCtor) {
          if (typeConstructors.count(maybeTypeCtor.value())) {
            argumentTypes.push_back(typeConstructors.lookup(maybeTypeCtor.value()).constructor(*this));
          } else {
            return emitError(loc(child)) << "Type constructor not found: " << maybeTypeCtor.value();
          }
        } else {
          argumentTypes.push_back(oboxType);
        }
      }

      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        Scope scope(symbolTable);
        builder.setInsertionPointToStart(module->getBody());
        auto funcType = builder.getFunctionType(argumentTypes, oboxType);
        auto func = builder.create<mlir::func::FuncOp>(loc(child), callee, funcType);
        func.setPrivate();
        auto *entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        assert(entryBlock->getArguments().size() == arguments.size());
        for (auto [arg, argName] : llvm::zip(entryBlock->getArguments(), arguments)) {
          if (failed(declareValue(argName.first, arg))) {
            return emitError(loc(child)) << "Failed to declare value: " << argName.first;
          }
        }
        mlir::Value lastValue;
        while (it != letChildren.end()) {
          lastValue = must(gen(it++));
        }
        auto castedValue =
            builder.createConvert(loc(child), lastValue, oboxType);
        builder.create<mlir::func::ReturnOp>(loc(child), castedValue);
      }
      return mlir::Value();
    }
  } else if (childType == "value_definition") {
    auto children = childrenNodes(child);
    auto it = children.begin();
    assert(it++->first == "let");
    assert(it->first == "let_binding");
    return gen(it);
  } else if (childType == "expression_item") {
    auto children = childrenNodes(child);
    return gen(children.begin());
  } else if (childType == "type_definition") {
    auto children = childrenNodes(child);
    return mlir::emitError(loc(child)) << "Type definition not implemented";
    // declareTypeConstructor("shape", [](MLIRGen &gen) {
    //   auto i64 = gen.builder.getI64Type();
    //   return mlir::ocaml::VariantType::get("shape", {i64});
    // });
  } else if (childType == "application_expression") {
    auto children = childrenNodes(child);
    auto it = children.begin();
    assert(it->first == "value_path");

    auto callee = must(valuePathToIdentifier(&it++->second));

    SmallVector<mlir::Value> args;
    while (it != children.end()) {
      args.push_back(must(gen(it++)));
      DBGS("arg: " << args.back() << "\n");
    }

    if (auto runtimeCall = genRuntimeCall(callee, args, loc(child), &child);
        succeeded(runtimeCall)) {
      return runtimeCall;
    } else if (auto func = lookupFunction(callee); succeeded(func)) {
      DBGS("calling function " << callee << " " << func->getFunctionType() << "\n");
      auto argTypes = func->getArgumentTypes();
      if (args.size() != argTypes.size()) {
        return emitError(loc(child))
               << "Function " << callee << " expects " << argTypes.size()
               << " arguments, but got " << args.size();
      }
      auto convertedArgs = llvm::to_vector(llvm::map_range(llvm::enumerate(args), [&](auto arg) {
        DBGS("converting arg type: " << arg.value().getType() << " to " << argTypes[arg.index()] << "\n");
        return builder.create<mlir::ocaml::ConvertOp>(loc(child), argTypes[arg.index()], arg.value()).getResult();
      }));
      auto call = builder.create<mlir::func::CallOp>(loc(child), *func, convertedArgs);
      return call.getResult(0);
    } else {
      return emitError(loc(child)) << "Function not found: " << callee;
    }
  } else if (childType == "number") {
    auto text = adaptor->text(&child);
    StringRef textRef = text;
    int result;
    textRef.getAsInteger(10, result);
    auto op = builder.create<mlir::arith::ConstantIntOp>(loc(child), result, 64);
    auto box = builder.createEmbox(loc(child), op);
    return box;
  } else if (childType == "parenthesized_expression") {
    auto children = childrenNodes(child);
    assert(children.size() == 3);
    return gen(children.begin() + 1);
  } else if (childType == "for_expression") {
    auto forChildren = childrenNodes(child);
    auto it = forChildren.begin();
    assert(it++->first == "for");
    assert(it->first == "value_pattern");
    auto maybeIterVar = valuePathToIdentifier(&it++->second);
    if (failed(maybeIterVar)) {
      return emitError(loc(child)) << "Expected iterator variable";
    }
    auto iterVar = maybeIterVar.value();
    assert(it++->first == "=");
    auto lowerBound = must(gen(it++));
    lowerBound = builder.createConvert(loc(child), lowerBound, builder.getI64Type());
    assert(it++->first == "to");
    auto upperBound = must(gen(it++));
    upperBound = builder.createConvert(loc(child), upperBound, builder.getI64Type());
    mlir::Value step = builder.create<mlir::arith::ConstantIntOp>(loc(child), 1, 64);
    auto loop = builder.create<mlir::scf::ForOp>(loc(child), lowerBound, upperBound, step);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      auto iterVarBlockArgument = loop.getInductionVar();
      must(declareValue(iterVar, iterVarBlockArgument));
      assert(it->first == "do_clause");
      auto doClauseChildren = childrenNodes(it->second);
      auto doIt = doClauseChildren.begin();
      assert(doIt++->first == "do");
      while (doIt->first != "done") {
        must(gen(doIt++));
      }
    }
    return mlir::Value();
  } else {
    llvm::errs() << "Unhandled node type: " << childType << "\n" << text << "\n";
    return emitError(loc(child)) << "Unhandled node type: " << childType;
  }
  return mlir::Value();
}

FailureOr<mlir::Value> MLIRGen::genAssign(StringRef lhs, mlir::Value rhs) {
  (void)lhs;
  (void)rhs;
  return mlir::Value();
}

FailureOr<mlir::Value> MLIRGen::gen(NodeList & nodes) {
  auto it = nodes.begin();
  FailureOr<mlir::Value> result;
  while (it != nodes.end()) {
    result = gen(it++);
    if (failed(result)) {
      return failure();
    }
  }
  return result;
}

void MLIRGen::insertBuiltinTypeConstructors() {
  auto *ctx = &getContext();
  typeConstructors.insert("int", TypeConstructor{
    .constructor = [](MLIRGen &gen) { return mlir::ocaml::BoxType::get(gen.builder.getI64Type()); },
  });
  typeConstructors.insert("float", TypeConstructor{
    .constructor = [](MLIRGen &gen) { return mlir::ocaml::BoxType::get(gen.builder.getF64Type()); },
  });
  typeConstructors.insert("string", TypeConstructor{
    .constructor = [ctx](MLIRGen &gen) { return mlir::ocaml::StringType::get(ctx); },
  });
  typeConstructors.insert("unit", TypeConstructor{
    .constructor = [ctx](MLIRGen &gen) { return mlir::ocaml::UnitType::get(ctx); },
  });
}

void MLIRGen::genCompilationUnit(TSNode node) {
  Scope scope(symbolTable);
  TypeConstructorScope typeConstructorScope(typeConstructors);
  insertBuiltinTypeConstructors();
  StringRef nodeType = ts_node_type(node);
  assert(nodeType == "compilation_unit");
  auto children = childrenNodes(node);

  mlir::Type returnType = builder.getI32Type();
  mlir::FunctionType funcType = builder.getFunctionType({}, returnType);
  mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);
  mlir::Block &entryBlock = funcOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);
  auto lastResult = gen(children);
  auto result = must(lastResult);
  if (not result) {
    result = builder.create<mlir::arith::ConstantIntOp>(loc(node), 0, 32);
  }
  if (result.getType() != returnType) {
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(loc(node),
                                                           returnType, result)
                 .getResult(0);
  }
  builder.create<mlir::func::ReturnOp>(loc(node), result);
}

FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen::gen(TSTreeAdaptor &&adaptor) {
  this->adaptor = &adaptor;
  auto filenameAttr = builder.getStringAttr(adaptor.getFilename());
  module = mlir::ModuleOp::create(mlir::FileLineColLoc::get(filenameAttr, 0, 0));
  builder.setInsertionPointToEnd(module->getBody());
  auto root = ts_tree_root_node(adaptor);
  genCompilationUnit(root);
  return std::move(module);
}
