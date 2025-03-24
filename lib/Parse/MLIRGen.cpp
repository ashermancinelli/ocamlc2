#include "ocamlc2/Parse/MLIRGen.h"
#include <tree_sitter/api.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Attributes.h>
#include "ocamlc2/Parse/TSAdaptor.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

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

FailureOr<mlir::Value> MLIRGen::gen(Node node) {
  auto [childType, child] = node;
  if (childType == "number") {
    long long ival = std::stoll(adaptor->text(child));
    auto iType = builder.getIntegerType(64);
    auto iAttr = mlir::IntegerAttr::get(iType, ival);
    auto val = builder.create<mlir::arith::ConstantOp>(loc(child), iAttr);
    return val.getResult();
  }
  assert(false && "Unhandled node type");
  return mlir::Value();
}

FailureOr<mlir::Value> MLIRGen::gen(NodeIter &it) {
  auto [childType, child] = *it++;
  llvm::dbgs() << "gen: " << childType << "\n";
  if (childType == "comment") {
    return mlir::Value();
  } else if (childType == "let") {
    auto [lbType, lb] = *it++;
    assert(lbType == "let_binding");
    auto letChildren = childrenNodes(lb);

    assert(letChildren[1].first == "=");
    mlir::Value rhsValue = must(gen(letChildren[2]));
    mlir::Value lhsValue = must(genAssign(letChildren[0], rhsValue));
    return lhsValue;
  } else if (childType == "for_expression") {
    auto forChildren = childrenNodes(child);
    auto iterArg = forChildren[1];
    auto lb = forChildren[3], ub = forChildren[5];
    auto body = forChildren[6];
    assert(body.first == "do_clause");
  } else {
    assert(false && "Unhandled node type");
  }
  return mlir::Value();
}

FailureOr<mlir::Value> MLIRGen::genAssign(Node lhs, mlir::Value rhs) {
  (void)lhs;
  (void)rhs;
  return mlir::Value();
}

FailureOr<mlir::Value> MLIRGen::gen(NodeList & nodes) {
  auto it = nodes.begin();
  while (it != nodes.end()) {
    auto [childType, child] = *it++;
    auto maybeValue = gen(it);
    ++it;
    if (failed(maybeValue)) {
      return failure();
    }
  }
  return mlir::Value();
}

void MLIRGen::genCompilationUnit(TSNode node) {
  StringRef nodeType = ts_node_type(node);
  assert(nodeType == "compilation_unit");
  auto children = childrenNodes(node);
  (void)gen(children);
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
