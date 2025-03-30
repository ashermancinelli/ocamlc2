#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlPasses.h"

namespace mlir::ocaml {

#define GEN_PASS_DEF_BUFFERIZEBOXES
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

namespace {

class BufferizeBoxesRewriter : public OpRewritePattern<mlir::ocaml::ConvertOp> {
public:
  using OpRewritePattern<mlir::ocaml::ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::ocaml::ConvertOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};


struct BufferizeBoxes : public impl::BufferizeBoxesBase<BufferizeBoxes> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<BufferizeBoxesRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
}
}


