#include "ocamlc2/Parse/MLIRGen.h"
#include <tree_sitter/api.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Attributes.h>
#include "ocamlc2/Parse/TSAdaptor.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
MLIRGen::MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder)
    : context(context), builder(builder) {
}

static std::vector<std::pair<StringRef, TSNode>> childrenNodes(TSNode node) {
  unsigned child_count = ts_node_child_count(node);
  std::vector<std::pair<StringRef, TSNode>> children;
  for (unsigned i = 0; i < child_count; ++i) {
    TSNode child = ts_node_child(node, i);
    children.emplace_back(ts_node_type(child), child);
  }
  return children;
}

mlir::Location MLIRGen::loc(TSNode node) {
  auto pt = ts_node_start_point(node);
  return mlir::FileLineColLoc::get(builder.getStringAttr(adaptor->getFilename()), pt.row, pt.column);
}

FailureOr<mlir::Value> MLIRGen::gen(NodeIter it) {
  auto [childType, child] = *it;
  llvm::dbgs() << "gen: " << childType << "\n";
  if (childType == "comment") {
    return mlir::Value();
  } else if (childType == "let_binding") {
    auto letChildren = childrenNodes(child);
    assert(letChildren.size() == 3);
    auto rhs = must(gen(letChildren.begin() + 2));
    if (letChildren[0].first == "unit") {
      return rhs;
    }
    mlir::Value lhsValue = must(genAssign(letChildren[0].first, rhs));
    return lhsValue;
  } else if (childType == "value_definition") {
    auto children = childrenNodes(child);
    auto it = children.begin();
    assert(it++->first == "let");
    assert(it->first == "let_binding");
    return gen(it);
  } else if (childType == "application_expression") {
    auto children = childrenNodes(child);
    auto it = children.begin();
    assert(it->first == "value_path");
    StringRef callee = "printf";

    // auto calleeNode = must(gen(it++));
    // auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto printfFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, true);
    // auto one = builder.create<mlir::arith::ConstantIntOp>(loc(child), 1, 64);
    // auto ptr = builder.create<mlir::LLVM::AllocaOp>(loc(child), ptrType, one.getResult());
    mlir::LLVM::LLVMFuncOp printfFunc;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module->getBody());
      printfFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
          loc(child), callee, printfFuncType);
    }
    auto printfCall = builder.create<mlir::LLVM::CallOp>(loc(child), printfFunc, mlir::ValueRange{});
    return printfCall.getResult();
  } else if (childType == "number") {
    auto text = adaptor->text(child);
    StringRef textRef = text;
    int result;
    textRef.getAsInteger(10, result);
    auto op = builder.create<mlir::arith::ConstantIntOp>(loc(child), result, 64);
    return op.getResult();
  } else if (childType == "for_expression") {
    auto forChildren = childrenNodes(child);
    auto it = forChildren.begin();
    assert(it++->first == "for");
    assert(it->first == "value_pattern");
    StringRef iterVar = it++->first;
    (void)iterVar;
    assert(it++->first == "=");
    assert(it->first == "number");
    auto lowerBound = must(gen(it++));
    assert(it++->first == "to");
    assert(it->first == "number");
    auto upperBound = must(gen(it++));
    mlir::Value step = builder.create<mlir::arith::ConstantIntOp>(loc(child), 1, 64);
    auto loop = builder.create<mlir::scf::ForOp>(loc(child), lowerBound, upperBound, step);
    mlir::Value result;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      assert(it->first == "do_clause");
      auto doClauseChildren = childrenNodes(it->second);
      auto doIt = doClauseChildren.begin();
      assert(doIt++->first == "do");
      while (doIt->first != "done") {
        result = must(gen(doIt++));
      }
    }
    return upperBound;
  } else {
    llvm::errs() << "Unhandled node type: " << childType << "\n";
    assert(false && "Unhandled node type");
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

void MLIRGen::genCompilationUnit(TSNode node) {
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
