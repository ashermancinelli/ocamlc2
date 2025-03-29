#ifndef OCAML_C2_DIALECT_TD
#define OCAML_C2_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ocaml_Dialect : Dialect {
  let name = "ocaml";
  let cppNamespace = "::mlir::ocaml";
  // let useDefaultTypePrinterParser = 1;
}

class Ocaml_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ocaml_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ocaml_BoxType : Ocaml_Type<"box", "box"> {
  let summary = "OCaml box";
  let description = "OCaml box";
  let parameters = (ins Optional<AnyType>:$elementType);
  let assemblyFormat = "(`<` type($elementType) `>` $elementType^)?";
}

class Ocaml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ocaml_Dialect, mnemonic, traits>;

def Ocaml_CastOp : Ocaml_Op<"cast"> {
  let summary = "OCaml cast";
  let description = "OCaml cast";
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

#endif
