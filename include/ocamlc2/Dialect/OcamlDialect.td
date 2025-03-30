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

def AnyBoxType : TypeConstraint<CPred<"mlir::isa<mlir::ocaml::BoxType, mlir::ocaml::OpaqueBoxType>($_self)">>;

class Ocaml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ocaml_Dialect, mnemonic, traits>;

def Ocaml_IntrinsicOp : Ocaml_Op<"intrinsic"> {
  let summary = "OCaml intrinsic";
  let description = "OCaml intrinsic";
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$callee attr-dict `(` $args `)` `:` type($args) `->` type($result)";
}

def Ocaml_ConvertOp : Ocaml_Op<"convert"> {
  let summary = "OCaml convert";
  let description = "OCaml convert";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
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
