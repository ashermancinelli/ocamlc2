
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#define DEBUG_TYPE "type-inference"
#include "ocamlc2/Support/Debug.h.inc"

namespace mlir::ocaml {

#define GEN_PASS_DEF_TYPEINFERENCE
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

namespace {

struct TypeInferencePattern : public OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern<mlir::scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::IfOp op,
                               PatternRewriter &rewriter) const override {
    DBGS("Running type inference on: " << op << "\n");
    auto resultType = op->getResultTypes();
    auto context = op->getContext();
    if (!(op->hasAttr(getMatchCaseAttr(context).getName()) &&
          mlir::isa<mlir::ocaml::OpaqueBoxType>(resultType[0]))) {
      return failure();
    }
    SmallVector<mlir::scf::IfOp> ifOps;
    op->walk([&](mlir::scf::IfOp ifOp) {
      ifOps.push_back(ifOp);
    });
    // Work inside-out to rewrite the types. Assume the innermost ifOp
    // converts the true type of the match expression into an opaque box,
    // and that the rest of the branches will match it after the rewrite.
    for (auto ifOp : llvm::reverse(ifOps)) {
      auto &thenRegion = ifOp.getThenRegion();
      auto *yieldOp = thenRegion.back().getTerminator();
      auto converted = yieldOp->getOperand(0);
      auto convertedType = converted.getType();
      if (convertedType != resultType[0]) {
        yieldOp->setOperand(0, converted);
      }
    }
    return success();
  }
};

struct TypeInference : public impl::TypeInferenceBase<TypeInference> {
  void runOnOperation() override {
    DBGS("Running TypeInference pass\n");
    
    RewritePatternSet patterns(&getContext());
    patterns.add<TypeInferencePattern>(&getContext());
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::ocaml
