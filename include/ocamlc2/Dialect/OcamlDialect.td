#ifndef OCAML_C2_DIALECT_TD
#define OCAML_C2_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def Ocaml_Dialect : Dialect {
  let name = "ocaml";
  let cppNamespace = "::mlir::ocaml";
  let useDefaultTypePrinterParser = 1;
}

class Ocaml_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ocaml_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ocaml_VariantType : Ocaml_Type<"Variant", "variant"> {
  let summary = "OCaml variant";
  let description = "OCaml variant";
  let parameters = (ins "mlir::StringAttr":$name, ArrayRefParameter<"mlir::Type">: $elements);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::StringRef":$name,
      "llvm::ArrayRef<mlir::Type>":$members), [{
      auto nameAttr = mlir::StringAttr::get(members[0].getContext(), name);
      return $_get(members[0].getContext(), nameAttr, members);    
    }]>,
    TypeBuilderWithInferredContext<(ins
      "mlir::StringAttr":$name,
      "llvm::ArrayRef<mlir::Type>":$members), [{
      return $_get(name.getContext(), name, members);    
    }]>,
  ];
  let extraClassDeclaration = [{
    enum Kind { Variant, Struct };
    std::string getNameString() const {
      return name.str();
    }
  }];
}

def Ocaml_BoxType : Ocaml_Type<"Box", "box"> {
  let summary = "OCaml box";
  let description = "OCaml box";
  let parameters = (ins AnyType:$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
  let assemblyFormat = "`<` $elementType `>`";
}

def Ocaml_OpaqueBoxType : Ocaml_Type<"OpaqueBox", "obox"> {
  let summary = "OCaml opaque box";
  let description = "OCaml opaque box";
  let parameters = (ins);
  let assemblyFormat = "";
}

def Ocaml_StringType : Ocaml_Type<"String", "sbox"> {
  let summary = "OCaml string";
  let description = "OCaml string";
  let parameters = (ins);
  let assemblyFormat = "";
}

def Ocaml_UnitType : Ocaml_Type<"Unit", "unit"> {
  let summary = "OCaml unit";
  let description = "OCaml unit";
  let parameters = (ins);
  let assemblyFormat = "";
}

def AnyBoxType : TypeConstraint<CPred<"mlir::isa<mlir::ocaml::BoxType, mlir::ocaml::OpaqueBoxType, mlir::ocaml::UnitType>($_self)">>;

class Ocaml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ocaml_Dialect, mnemonic, traits>;

def Ocaml_UnitOp : Ocaml_Op<"unit"> {
  let summary = "OCaml unit";
  let description = "OCaml unit";
  let arguments = (ins);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let results = (outs Ocaml_UnitType:$result);
  let assemblyFormat = "attr-dict";
}

def Ocaml_IntrinsicOp : Ocaml_Op<"builtin"> {
  let summary = "OCaml intrinsic";
  let description = "OCaml intrinsic";
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($args) `->` type($result)";
}

def Ocaml_ConvertOp : Ocaml_Op<"convert"> {
  let summary = "OCaml convert";
  let description = "OCaml convert";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let extraClassDeclaration = [{
    mlir::Type getFromType() {
      return getInput().getType();
    }
    mlir::Type getToType() {
      return getType();
    }
  }];
  let hasFolder = 1;
  let assemblyFormat = "$input attr-dict `from` type($input) `to` type($result)";
}

def Ocaml_PrintfOp : Ocaml_Op<"printf"> {
    let summary = "OCaml printf";
    let description = "OCaml printf";
    let arguments = (ins Variadic<AnyType>:$args);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$args attr-dict `:` type($args) `->` type($result)";
}

def Ocaml_ObjReprOp : Ocaml_Op<"repr"> {
    let summary = "OCaml repr";
    let description = "OCaml repr";
    let arguments = (ins AnyBoxType:$input);
    let results = (outs I32:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif
