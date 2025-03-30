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
      auto printfOp = builder.create<mlir::ocaml::PrintfOp>(loc, builder.getI32Type(), args);
      return printfOp.getResult();
    }});
    rtfs.push_back({"float", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      auto floatType = mlir::ocaml::BoxType::get(builder.getF64Type());
      auto boxedFloat = builder.createConvert(loc, args[0], floatType);
      return boxedFloat;
    }});
    rtfs.push_back({"Obj.repr", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      auto reprOp = builder.create<mlir::ocaml::ObjReprOp>(loc, builder.getI32Type(), args[0]);
      return reprOp.getResult();
    }});
    rtfs.push_back({"print_float", [](MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args) -> mlir::Value {
      auto &builder = gen->getBuilder();
      auto module = gen->getModule();
      auto oboxType = mlir::ocaml::OpaqueBoxType::get(builder.getContext());
      auto llptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
      if (not module.lookupSymbol("print_float")) {
        auto insertionGuard =
            mlir::OpBuilder::InsertionGuard(gen->getBuilder());
        builder.setInsertionPointToStart(module.getBody());
        auto printFloatFuncType = mlir::FunctionType::get(builder.getContext(), {llptrType}, {llptrType});
        auto func = builder.create<mlir::func::FuncOp>(loc, "print_float", printFloatFuncType);
        func.setPrivate();
      }
      assert(args.size() == 1);
      auto arg = builder.createConvert(loc, args[0], llptrType);
      auto printFloatOp = builder.create<mlir::func::CallOp>(
          loc, "print_float", mlir::TypeRange{llptrType},
          mlir::ValueRange{arg});
      auto result = builder.createConvert(loc, printFloatOp.getResult(0), oboxType);
      return result;
    }});
  }
  return rtfs;
}
