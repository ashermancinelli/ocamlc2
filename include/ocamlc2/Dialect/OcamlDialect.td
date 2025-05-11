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

def Ocaml_TupleType : Ocaml_Type<"Tuple", "tuple"> {
  let summary = "OCaml tuple";
  let description = "OCaml tuple";
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<` $types `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<::mlir::Type>":$types), [{
      return $_get(types[0].getContext(), types);
    }]>,
  ];
}

def Ocaml_VariantType : Ocaml_Type<"Variant", "variant"> {
  let summary = "OCaml variant";
  let description = "OCaml variant";
  let parameters = (ins 
    "::mlir::StringAttr":$name,
    ArrayRefParameter<"::mlir::StringAttr">:$constructors,
    ArrayRefParameter<"::mlir::Type">:$types
  );
  let storageClass = "VariantTypeStorage";
  let storageNamespace = "detail";
  let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::StringAttr":$name,
      "::llvm::ArrayRef<::mlir::StringAttr>":$names,
      "::llvm::ArrayRef<::mlir::Type>":$types), [{
      return $_get(name.getContext(), name, names, types);
    }]>,
  ];
  let extraClassDeclaration = [{
    enum Kind { Variant, Record };
    std::string getNameString() const {
      return name.str();
    }
    static mlir::FailureOr<std::pair<unsigned, mlir::Type>> typeForConstructor(llvm::StringRef name, VariantType type);
  }];
}

def Ocaml_BoxType : Ocaml_Type<"Box", "box"> {
  let summary = "OCaml box";
  let description = "OCaml box";
  let parameters = (ins AnyType:$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
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

def AnyBoxType : TypeConstraint<CPred<"::mlir::isa<::mlir::ocaml::BoxType, ::mlir::ocaml::OpaqueBoxType, ::mlir::ocaml::UnitType>($_self)">>;

def AnyOCamlType : TypeConstraint<
  CPred<"::mlir::isa<"
        "::mlir::ocaml::BoxType, ::mlir::ocaml::OpaqueBoxType, "
        "::mlir::ocaml::StringType, ::mlir::ocaml::UnitType, "
        "::mlir::ocaml::TupleType, ::mlir::ocaml::VariantType>($_self)">,
  "OCaml value">;

class Ocaml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ocaml_Dialect, mnemonic, traits>;

def Ocaml_EmboxStringOp : Ocaml_Op<"embox_string"> {
  let summary = "OCaml embox string";
  let description = "OCaml embox string";
  let arguments = (ins StrAttr:$input);
  let results = (outs Ocaml_StringType:$result);
  let assemblyFormat = "$input attr-dict";
}

def Ocaml_PatternVariableOp : Ocaml_Op<"pattern_variable"> {
  let summary = "OCaml pattern variable";
  let description = "OCaml pattern variable";
  let arguments = (ins);
  let results = (outs AnyOCamlType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ocaml_PatternMatchOp : Ocaml_Op<"match", [Pure, SameTypeOperands]> {
  let summary = "OCaml pattern match";
  let description = "OCaml pattern match";
  let arguments = (ins AnyOCamlType:$scrutinee, AnyOCamlType:$pattern);
  let results = (outs I1:$result);
  let assemblyFormat = "$scrutinee `against` $pattern attr-dict `:` type($scrutinee)";
}

def Ocaml_TupleCtorOp : Ocaml_Op<"tuple_ctor"> {
  let summary = "OCaml tuple constructor";
  let description = "OCaml tuple constructor";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Ocaml_TupleType:$result);
  let assemblyFormat = "$elements attr-dict `:` type($elements) `->` type($result)";
}

def Ocaml_TupleGetOp : Ocaml_Op<"tuple_get"> {
  let summary = "OCaml tuple get";
  let description = "OCaml tuple get";
  let arguments = (ins Ocaml_TupleType:$tuple, I64:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tuple `[` $index `]` attr-dict `:` type($tuple) `,` type($index) `->` type($result)";
}

def Ocaml_UnitOp : Ocaml_Op<"unit"> {
  let summary = "OCaml unit";
  let description = "OCaml unit";
  let arguments = (ins);
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
