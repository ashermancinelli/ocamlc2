#ifndef OCAML_C2_DIALECT_TD
#define OCAML_C2_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
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
  let assemblyFormat = "`<` $types `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<::mlir::Type>":$types), [{
      return $_get(types[0].getContext(), types);
    }]>,
  ];
}

def Ocaml_ReferenceType : Ocaml_Type<"Reference", "ref"> {
  let summary = "OCaml reference";
  let description = "OCaml reference";
  let parameters = (ins AnyType:$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
}

def Ocaml_ArrayType : Ocaml_Type<"Array", "array"> {
  let summary = "OCaml array";
  let description = "OCaml array";
  let parameters = (ins AnyType:$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
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

def Ocaml_EnvType : Ocaml_Type<"Env", "env"> {
  let summary = "OCaml environment";
  let description = "OCaml environment for closures and captured values";
  let parameters = (ins);
  let assemblyFormat = "";
}

def Ocaml_ClosureType : Ocaml_Type<"Closure", "closure"> {
  let summary = "OCaml closure";
  let description = "OCaml closure";
  let parameters = (ins FunctionType:$functionType);
  let assemblyFormat = "`<` $functionType `>`";
}

def AnyBoxType : TypeConstraint<
  CPred<"::mlir::isa<"
        "::mlir::ocaml::BoxType, "
        "::mlir::ocaml::OpaqueBoxType, "
        "::mlir::ocaml::UnitType>($_self)">,
  "OCaml box">;

def AnyOCamlType : TypeConstraint<
  CPred<"::mlir::isa<"
        "::mlir::ocaml::BoxType, ::mlir::ocaml::OpaqueBoxType, "
        "::mlir::ocaml::StringType, ::mlir::ocaml::UnitType, "
        "::mlir::ocaml::TupleType, ::mlir::ocaml::VariantType, "
        "::mlir::ocaml::ArrayType>($_self)">,
  "OCaml value">;

def AnyOcamlReferenceType : TypeConstraint<
  CPred<"::mlir::isa<::mlir::ocaml::ReferenceType>($_self)">,
  "OCaml reference">;

class Ocaml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ocaml_Dialect, mnemonic, traits>;

def Ocaml_EnvOp : Ocaml_Op<"closure.env.new", [Pure]> {
  let summary = "OCaml environment";
  let description = "OCaml environment for closures and captured values";
  let arguments = (ins);
  let results = (outs Ocaml_EnvType:$result);
  let assemblyFormat = "attr-dict";
}

def Ocaml_EnvCaptureOp : Ocaml_Op<"closure.env.capture"> {
  let summary = "OCaml environment capture";
  let description = "OCaml environment capture";
  let arguments = (ins Ocaml_EnvType:$env, StrAttr:$id, AnyOCamlType:$value);
  let results = (outs);
  let assemblyFormat = "$env `[` $id `]` `=` $value attr-dict `:` type($value)";
}

def Ocaml_EnvGetOp : Ocaml_Op<"closure.env.get", [MemoryEffectsOpInterface]> {
  let summary = "OCaml environment get";
  let description = "OCaml environment get";
  let arguments = (ins Arg<Ocaml_EnvType, "environment being read from", [MemRead]>:$env, StrAttr:$id);
  let results = (outs AnyOCamlType:$result);
  let assemblyFormat = "$env `[` $id `]` attr-dict `->` type($result)";
}

def Ocaml_ClosureOp : Ocaml_Op<"closure.new", [Pure]> {
  let summary = "OCaml closure";
  let description = "OCaml closure";
  let arguments = (ins FlatSymbolRefAttr:$symbol, Ocaml_EnvType:$env);
  let results = (outs Ocaml_BoxType:$result);
  let assemblyFormat = "$symbol `capturing` $env attr-dict `:` type($result)";
}

def Ocaml_InvokeOp : Ocaml_Op<"closure.invoke", [
    CallOpInterface]> {
  let summary = "OCaml invoke";
  let description = "OCaml invoke";
  let arguments = (ins 
    Ocaml_BoxType:$closure,
    Variadic<AnyType>:$args,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs AnyOCamlType:$result);
  let assemblyFormat = "$closure `(` $args `)` attr-dict `:` `(` type($closure) `)` `->` type($args) `->` type($result)";
  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return getClosure().getType().getElementType().cast<FunctionType>();
    }
    MutableOperandRange getArgOperandsMutable() { return getArgsMutable(); }
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }
    operand_range getArgOperands() { return {arg_operand_begin(), arg_operand_end()}; }
    CallInterfaceCallable getCallableForCallee() { return getClosure(); }
    void setCalleeFromCallable(CallInterfaceCallable callee) { return setOperand(0, cast<Value>(callee)); }
  }];
}

def Ocaml_LoadOp : Ocaml_Op<"load"> {
  let summary = "OCaml load";
  let description = "OCaml load";
  let arguments = (ins AnyOcamlReferenceType:$input);
  let results = (outs AnyOCamlType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Ocaml_StoreOp : Ocaml_Op<"store"> {
  let summary = "OCaml store";
  let description = "OCaml store";
  let arguments = (ins AnyOCamlType:$value, AnyOcamlReferenceType:$input);
  let results = (outs);
  let assemblyFormat = "$value `to` $input attr-dict `:` type($value) `to` type($input)";
}

def Ocaml_AddressOfOp : Ocaml_Op<"address_of"> {
  let summary = "OCaml address of";
  let description = "OCaml address of";
  let arguments = (ins SymbolRefAttr:$symbol);
  let results = (outs AnyOcamlReferenceType:$result);
  let assemblyFormat = "$symbol attr-dict `:` type($result)";
}

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

def Ocaml_ArrayMakeOp : Ocaml_Op<"array.make"> {
  let summary = "OCaml array make";
  let description = "OCaml array make";
  let arguments = (ins AnyOCamlType:$elementType);
  let results = (outs Ocaml_ArrayType:$result);
  let assemblyFormat = "$elementType attr-dict `:` type($elementType) `->` type($result)";
}

def Ocaml_ArrayGetOp : Ocaml_Op<"array.get"> {
  let summary = "OCaml array get";
  let description = "OCaml array get";
  let arguments = (ins Ocaml_ArrayType:$array, I64:$index);
  let results = (outs AnyOCamlType:$result);
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `,` type($index) `->` type($result)";
}

def Ocaml_ArrayFromElementsOp : Ocaml_Op<"array.from_elements"> {
  let summary = "OCaml array from elements";
  let description = "OCaml array from elements";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Ocaml_ArrayType:$result);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$elements), [{
      mlir::Type elementType = elements.front().getType();
      mlir::Type arrayType = mlir::ocaml::ArrayType::get(elementType.getContext(), elementType);
      return build($_builder, $_state, arrayType, elements);
    }]>,
  ];
  let assemblyFormat = "`[` $elements `]` attr-dict `:` type($elements) `->` type($result)";
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

def Ocaml_GlobalOp : Ocaml_Op<"global", [Symbol]> {
  let summary = "OCaml global variable (closure capture slot)";
  let description = [{
    Represents a single module-level global value that is used to materialise
    closure captures.  The op only declares the storage; actual reads/writes are
    performed with `ocaml.global_load` / `ocaml.global_store`.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    SymbolRefAttr:$symref,
    TypeAttr:$type
  );
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "::mlir::StringRef":$name, "mlir::Type":$type,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
  ];
  let extraClassDeclaration = [{
    mlir::Type resultType();
    mlir::FlatSymbolRefAttr getSymbol() {
      return mlir::FlatSymbolRefAttr::get(getContext(),
          (*this)->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()).getValue());
    }
  }];
}

def OcamlGlobalOp : TypeConstraint<
  CPred<"::mlir::isa<::mlir::ocaml::GlobalOp>($_self)">,
  "OCaml global">;

def Ocaml_GlobalLoadOp : Ocaml_Op<"global_load", [Pure]> {
  let summary = "Read from an ocaml.global";
  let description = "Returns the value stored in the given global symbol";
  let arguments = (ins SymbolRefAttr:$global);
  let results   = (outs AnyOCamlType:$result);
  let assemblyFormat = "$global attr-dict `:` type($result)";
}

def Ocaml_GlobalStoreOp : Ocaml_Op<"global_store", []> {
  let summary = "Write to an ocaml.global";
  let description = "Stores the operand into the given global symbol";
  let arguments = (ins AnyOCamlType:$value, SymbolRefAttr:$global);
  let results   = (outs);
  let assemblyFormat = "$value `to` $global attr-dict `:` type($value)";
}

#endif // OCAML_C2_DIALECT_TD
