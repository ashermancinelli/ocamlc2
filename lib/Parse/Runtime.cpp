#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <ocamlc2/Parse/Runtime.h>
#include <ocamlc2/Parse/MLIRGen.h>
#include <ocamlc2/Dialect/OcamlDialect.h>

mlir::Value RTPrintfCall(MLIRGen *gen, TSNode *node, mlir::Location loc,
                         mlir::ValueRange args) {
  auto module = gen->getModule();
  auto &builder = gen->getBuilder();
  auto opaquePtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  if (not module.lookupSymbol("printf")) {
    auto insertionGuard = mlir::OpBuilder::InsertionGuard(gen->getBuilder());
    builder.setInsertionPointToStart(module.getBody());
    auto printfFuncType = mlir::LLVM::LLVMFunctionType::get(
        builder.getI32Type(), {opaquePtrType}, true);
    builder.create<mlir::LLVM::LLVMFuncOp>(loc, "printf", printfFuncType);
  }
  auto argTypes = llvm::to_vector<4>(
      llvm::map_range(args, [](mlir::Value arg) { return arg.getType(); }));
  std::vector<mlir::Type> argTypesInto;
  argTypesInto.push_back(opaquePtrType);
  auto children = childrenNodes(*node);
  assert(children[0].first == "value_path");
  assert(children[1].first == "string" &&
         "only accepting printf with string literal format argument");
  auto stringContentNode = childrenNodes(children[1].second)[1].second;
  auto typeHints = must(gen->getPrintfTypeHints(args, &stringContentNode));
  llvm::copy(typeHints, std::back_inserter(argTypesInto));
  auto convertedArgs = llvm::to_vector(
      llvm::map_range(llvm::zip(args, argTypesInto), [&](auto argAndType) {
        auto [arg, type] = argAndType;
        if (arg.getType() == type) {
          return arg;
        }
        auto cast = builder.create<mlir::ocaml::ConvertOp>(loc, type, arg);
        return cast.getResult();
      }));
  auto printfFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(),
                                                          argTypesInto, true);
  auto printfCall = builder.create<mlir::LLVM::CallOp>(loc, printfFuncType,
                                                       "printf", convertedArgs);
  return printfCall.getResult();
}

llvm::ArrayRef<RuntimeFunction> RuntimeFunction::getRuntimeFunctions() {
  static SmallVector<RuntimeFunction> rtfs;
  if (rtfs.empty()) {
    rtfs.push_back({"Printf.printf", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      return builder.createCallIntrinsic(loc, "Printf.printf", args, builder.getUnitType());
    }});
    rtfs.push_back({"int", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      return builder.createConvert(loc, args[0], builder.emboxType(builder.getI32Type()));
    }});
    rtfs.push_back({"float", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      return builder.createConvert(loc, args[0], builder.emboxType(builder.getF64Type()));
    }});
    rtfs.push_back({"Obj.repr", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      auto reprOp = builder.createCallIntrinsic(loc, "Obj.repr", args);
      return reprOp;
    }});
    rtfs.push_back({"print_float", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      return builder.createCallIntrinsic(loc, "print_float", args, builder.getUnitType());
    }});
    rtfs.push_back({"print_int", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      return builder.createCallIntrinsic(loc, "print_int", args, builder.getUnitType());
    }});
  }
  return rtfs;
}
