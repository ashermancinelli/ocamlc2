#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>

#define DEBUG_TYPE "lower-runtime"
#include "ocamlc2/Support/Debug.h.inc"

namespace mlir::ocaml {

#define GEN_PASS_DEF_LOWEROCAMLRUNTIME
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

namespace {

static FailureOr<func::CallOp>
createGenericRuntimeCall(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                         mlir::ModuleOp module, StringRef calleeSuffix,
                         mlir::Type resultType, mlir::ValueRange args) {
  std::string callee = "ocamlrt." + calleeSuffix.str();
  DBGS("callee: " << callee << "\n");
  for (auto arg : args) {
    DBGS("arg: " << arg << "\n");
  }
  DBGS("resultType: " << resultType << "\n");
  auto argTypes = llvm::to_vector(
      llvm::map_range(args, [=](auto arg) -> mlir::Type { return arg.getType(); }));
  if (not module.lookupSymbol(callee)) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), argTypes, resultType);
    auto funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), callee, funcType);
    funcOp.setPrivate();
  }
  auto func = module.lookupSymbol<mlir::func::FuncOp>(callee);
  auto call = rewriter.create<mlir::func::CallOp>(op->getLoc(), func, args);
  DBGS("call: " << call << "\n");
  return call;
}

class LowerOCamlRuntimeRewriter : public OpRewritePattern<mlir::ocaml::IntrinsicOp> {
public:
  using OpRewritePattern<mlir::ocaml::IntrinsicOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::ocaml::IntrinsicOp op,
                                PatternRewriter &rewriter) const final {
    auto callee = op.getCallee();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    if (callee == "print_float" or callee == "box_convert_i64_f64" or callee == "box_convert_f64_i64" or callee == "print_int") {
      SmallVector<mlir::Value> args{rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), llvmPointerType, op.getArgs()[0])};
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee, llvmPointerType, args);
      if (failed(newValue)) {
        return op.emitError("Failed to create runtime call: ") << callee;
      }
      rewriter.replaceOp(op, newValue->getResult(0));
      return success();
    } else if (callee == "embox_string") {
      auto stringType = mlir::ocaml::StringType::get(rewriter.getContext());
      SmallVector<mlir::Value> args{op.getArgs()};
      if (args.size() != 1) {
        return op.emitError("embox_string expects 1 argument, got ") << args.size();
      }
      args[0] = rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), llvmPointerType, args[0]);
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee, llvmPointerType, args);
      if (succeeded(newValue)) {
        auto castedResult = rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), stringType, newValue->getResult(0));
        rewriter.replaceOp(op, castedResult.getResult());
        return success();
      }
      return op.emitError("Failed to create runtime call: ") << callee;
    } else if (callee == "embox_i64") {
      auto i64Type = rewriter.getI64Type();
      auto boxType = mlir::ocaml::BoxType::get(i64Type);
      SmallVector<mlir::Value> args{rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), i64Type, op.getArgs()[0])};
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee, llvmPointerType, args);
      if (failed(newValue)) {
        return op.emitError("Failed to create runtime call: ") << callee;
      }
      auto castedResult = rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), boxType, newValue->getResult(0));
      rewriter.replaceOp(op, castedResult);
      return success();
    } else if (callee == "variant_ctor_empty" or callee == "variant_ctor") {
      auto variantType = op.getResult().getType();
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee, variantType, op.getArgs());
      if (failed(newValue)) {
        return op.emitError("Failed to create runtime call: ") << callee;
      }
      rewriter.replaceOp(op, newValue->getResult(0));
      return success();
    } else if (callee == "+" or callee == "-" or callee == "*" or callee == "/" or callee == "%") {
      auto lhs = op.getArgs()[0];
      auto lhsType = dyn_cast<mlir::ocaml::BoxType>(lhs.getType());
      auto rhs = op.getArgs()[1];
      auto rhsType = dyn_cast<mlir::ocaml::BoxType>(rhs.getType());
      if (!lhsType || !rhsType) {
        return op.emitError("Expected box types for intrinsic: ") << callee;
      }
      auto resultType = resolveTypes(lhsType, rhsType, op.getLoc());
      if (failed(resultType)) {
        return op.emitError("Failed to resolve types for intrinsic: ") << callee;
      }
      auto maybeCallee = mlir::ocaml::binaryOpToRuntimeName(callee.str(), op.getLoc());
      if (failed(maybeCallee)) {
        return op.emitError("Failed to get runtime name for intrinsic: ") << callee;
      }
      auto callee = *maybeCallee;
      auto lhsSuffix = getPODTypeRuntimeName(lhsType.getElementType());
      auto rhsSuffix = getPODTypeRuntimeName(rhsType.getElementType());
      if (failed(lhsSuffix) || failed(rhsSuffix)) {
        return op.emitError("Failed to get the runtime name for the operands of ") << callee;
      }
      callee += "_" + *lhsSuffix + "_" + *rhsSuffix;
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee,
                                               *resultType, {lhs, rhs});
      if (failed(newValue)) {
        return op.emitError("Failed to create runtime call: ") << callee;
      }
      rewriter.replaceOp(op, newValue->getResult(0));
      return success();
    } else {
      op.emitError("Unsupported intrinsic: ") << callee;
      assert(false && "Unsupported intrinsic");
      return failure();
    }
    return failure();
  }
};

class LowerOCamlConversions : public OpRewritePattern<mlir::ocaml::ConvertOp> {
public:
  using OpRewritePattern<mlir::ocaml::ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::ocaml::ConvertOp op,
                                PatternRewriter &rewriter) const final {
    if (op->getUses().empty()) {
      DBGS("no uses, remove op\n");
      rewriter.eraseOp(op);
      return success();
    }
    auto fromType = op.getFromType();
    auto toType = op.getToType();
    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    DBGS(op << "\n");
    if (fromType == toType) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    auto maybeFromBoxType = mlir::dyn_cast<mlir::ocaml::BoxType>(fromType);
    auto maybeToBoxType = mlir::dyn_cast<mlir::ocaml::BoxType>(toType);

    if ((ocaml::isa_box_type(fromType) && llvmPointerType == toType) ||
        (ocaml::isa_box_type(toType) && fromType == llvmPointerType)) {
      DBGS("bitcast box to/from pointer, leave alone for now\n");
      return failure();
    }

    if (maybeFromBoxType && maybeToBoxType) {
      if (maybeFromBoxType.getElementType() == maybeToBoxType.getElementType()) {
        rewriter.replaceOp(op, op.getInput());
        return success();
      } else {
        std::string callee = "box_convert_";
        auto maybeFromType = getPODTypeRuntimeName(maybeFromBoxType.getElementType());
        auto maybeToType = getPODTypeRuntimeName(maybeToBoxType.getElementType());
        if (failed(maybeFromType) || failed(maybeToType)) {
          DBGS("failed to get runtime name for box conversion: " << fromType << " to " << toType << "\n");
          assert(false && "Unsupported type for box conversion");
          return failure();
        }
        callee += *maybeFromType + "_" + *maybeToType;
        rewriter.replaceOpWithNewOp<mlir::ocaml::IntrinsicOp>(op, maybeToBoxType, callee, op.getInput());
        return success();
      }
    } else if (maybeToBoxType) {
      DBGS("to box\n");
      auto elementType = maybeToBoxType.getElementType();
      if (elementType == fromType) {
        // if the source type is not a box and the destination type is a box, we
        // need to embox the source type.
        DBGS("simple embox\n");
        auto suffix = getPODTypeRuntimeName(elementType);
        if (failed(suffix)) {
          DBGS("failed to get runtime name for embox: " << elementType << "\n");
          assert(false && "Unsupported type for embox");
          return failure();
        }
        auto callee = "embox_" + *suffix;
        auto newOp = rewriter.replaceOpWithNewOp<mlir::ocaml::IntrinsicOp>(op, maybeToBoxType, callee, op.getInput());
        DBGS("newOp: " << newOp << "\n");
        return success();
      } else {
        DBGS("complex embox, needs two conversions\n");
        auto emboxedFromType = mlir::ocaml::BoxType::get(fromType);
        // Otherwise, insert a convertion that will become an embox, and then
        // convert between the two box types.
        auto step1Convert = rewriter.create<ConvertOp>(op.getLoc(), emboxedFromType, op.getInput());
        DBGS("step1Convert: " << step1Convert << "\n");
        auto step2Convert = rewriter.create<ConvertOp>(op.getLoc(), maybeToBoxType, step1Convert);
        DBGS("step2Convert: " << step2Convert << "\n");
        rewriter.replaceOp(op, step2Convert);
        return success();
      }
    } else {
      DBGS("neither are concrete boxes\n");
      if (ocaml::isa_box_type(fromType) && ocaml::isa_box_type(toType)) {
        // this is generally ok
        return failure();
      }
      DBGS("failed to get runtime name for conversion: " << fromType << " to " << toType << "\n");
      assert(false && "Unsupported type for conversion");
      return failure();
    }

    return failure();
  }
};

struct LowerOCamlRuntime : public impl::LowerOCamlRuntimeBase<LowerOCamlRuntime> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerOCamlRuntimeRewriter, LowerOCamlConversions>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
}
}


