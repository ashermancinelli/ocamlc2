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

#define DEBUG_TYPE "ocaml-lower-runtime"
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
    if (callee == "print_float" or callee == "box_convert_i64_f64" or callee == "box_convert_f64_i64") {
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
    } else if (callee == "Printf.printf") {
      SmallVector<mlir::Value> argsFrom = op.getArgs();
      SmallVector<mlir::Value> argsTo{rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), llvmPointerType, argsFrom[0])};
      auto it = argsFrom.begin();
      it++;
      while (it != argsFrom.end()) {
        argsTo.push_back(rewriter.create<mlir::ocaml::ConvertOp>(op.getLoc(), llvmPointerType, *it++));
      }
      auto newValue = createGenericRuntimeCall(rewriter, op, module, callee, llvmPointerType, argsTo);
      if (failed(newValue)) {
        return op.emitError("Failed to create runtime call: ") << callee;
      }
      rewriter.replaceOp(op, newValue->getResult(0));
      return success();
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
    } else {
      return op.emitError("Unsupported intrinsic: ") << callee;
    }
    return failure();
  }
};

static auto getPODTypeRuntimeName(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, FailureOr<std::string>>(type)
      .Case<mlir::Float64Type>([](auto) -> std::string { return "f64"; })
      .Case<mlir::IntegerType>([](auto) -> std::string { return "i64"; })
      .Default([](auto) -> LogicalResult { return failure(); });
}

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
    DBGS("fromType: " << fromType << "\n");
    DBGS("toType: " << toType << "\n");
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
          return op.emitError("Unsupported type for box conversion: ") << fromType << " to " << toType;
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
          return op.emitError("Unsupported type for embox: ") << elementType;
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
      return op.emitError("Unsupported type for conversion: ") << fromType << " to " << toType;
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


