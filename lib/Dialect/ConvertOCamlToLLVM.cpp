// ocaml lower to llvm
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

#define DEBUG_TYPE "ocaml-to-llvm"
#include "ocamlc2/Support/Debug.h.inc"

namespace mlir::ocaml {

#define GEN_PASS_DEF_CONVERTOCAMLTOLLVM
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

namespace {

class ConvertOCamlToLLVMRewriter
    : public ConversionPattern {
public:
  ConvertOCamlToLLVMRewriter(LLVMTypeConverter &typeConverter,
                             MLIRContext *context)
      : ConversionPattern(typeConverter, "ocaml.convert", /*benefit=*/10, context) {}

  // Match against any ocaml.convert operation
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    DBGS("Found ocaml.convert op: " << *op << "\n");

    // Ensure there's exactly one operand
    if (operands.size() != 1) {
       return rewriter.notifyMatchFailure(op, "expected one operand");
    }
    Value operand = operands[0]; // Already converted type

    // Get the result type from the operation
    Type resultType = op->getResultTypes()[0];
    Type targetType = typeConverter->convertType(resultType);
    
    if (!targetType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    // Avoid inserting identity casts
    if (operand.getType() == targetType) {
      rewriter.replaceOp(op, operand);
      return success();
    } 
    
    // Create an unrealized conversion cast
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, targetType, operand);
    return success();
  }
};

static void addTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type type) -> std::optional<Type> {
    DBGS("type conversion: " << type << "\n");
    if (isa<OpaqueBoxType, BoxType,
                  UnitType, StringType,
                  VariantType, TupleType>(type)) {
      DBGS("OCaml type converted to LLVM pointer: " << type << "\n");
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    }
    if (type.getDialect().getNamespace() ==
            mlir::LLVM::LLVMDialect::getDialectNamespace() ||
        mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(type)) {
      DBGS("type is already legal: " << type << "\n");
      return type;
    }
    DBGS("type conversion failed/unhandled for: " << type << "\n");
    return std::nullopt;
  });

  typeConverter.addArgumentMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        DBGS("argument materialization: " << resultType << " " << loc << "\n");
        if (inputs.size() != 1)
          return nullptr;
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType,
          ValueRange inputs, Location loc) -> Value {
        DBGS("source materialization: " << resultType << " " << loc << "\n");
        for (Value input : inputs) {
          DBGS("input: " << input << "\n");
        }
        if (inputs.size() != 1)
          return nullptr;
        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType,
          ValueRange inputs, Location loc) -> Value {
        DBGS("target materialization: " << resultType << " " << loc << "\n");
        for (Value input : inputs) {
          DBGS("input: " << input << "\n");
        }
        if (inputs.size() != 1)
          return nullptr;
        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}

struct ConvertOCamlToLLVM
    : public impl::ConvertOCamlToLLVMBase<ConvertOCamlToLLVM> {
  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<OcamlDialect>();
    
    target.addIllegalOp<ConvertOp>();
    LLVMTypeConverter typeConverter(&getContext());
    addTypeConversions(typeConverter);

    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          // Only check the signature; the patterns handle body conversion.
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) {
          return typeConverter.isLegal(op.getOperandTypes());
        });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op);
    });

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter,
                                                         patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // Add our OCaml conversion pattern with high benefit to ensure it's applied first
    patterns.add<ConvertOCamlToLLVMRewriter>(typeConverter, &getContext());

    // Dump all conversion patterns for debugging
    DBGS("Available patterns:\n");
    for (const auto &pattern : patterns.getNativePatterns()) {
      DBGS("  Pattern: " << pattern->getRootKind() << "\n");
    }

    // Apply the patterns to the module
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::ocaml
