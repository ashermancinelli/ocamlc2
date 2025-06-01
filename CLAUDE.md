# CLAUDE.md

## Build and Test Commands

- **Build project**: `ninja -C build` or `make build`
- **Build specific tools**: `ninja -C build g3` for the code generator, `ninja -C build p3` for the parser
- **Run code generation tests**: `lit -svv test/cg`
- **Run type checking tests**: `lit -svv test/typecheck`
- **Run unit tests**: `ctest --test-dir build -VV`

## Key Tools

- **g3**: Main code generator (`./build/bin/g3 <file.ml>`)
- **p3**: Parser/REPL (`./build/bin/p3`)
- **ocaml-lsp-server**: Language server implementation
- **ocaml-opt**: Optimizer

## Architecture Overview

This is a toy OCaml compiler built on MLIR infrastructure that implements:

1. **Type System & Unification**: Hindley-Milner type inference with support for:
   - Polymorphic types and type variables
   - Function types with labeled/optional parameters
   - Variant types (sum types) and record types
   - Module types and functors
   - Recursive type definitions

2. **Multi-Stage Compilation Pipeline**:
   - **Tree-sitter parsing** → **Type inference** → **MLIR generation** → **LLVM lowering**
   - Uses custom OCaml MLIR dialect with operations for closures, variants, references, etc.

3. **Key Components**:
   - `Unifier` class: Implements type inference and unification algorithm
   - `MLIRGen3` class: Generates MLIR from typed AST
   - Custom MLIR dialect in `OcamlDialect.td` with OCaml-specific types and operations
   - Module system with signatures and functors

## Important Files

- **Type System**: `lib/Parse/Unifier/` contains the core type inference engine
- **MLIR Generation**: `lib/Parse/MLIRGen3.cpp` translates typed AST to MLIR
- **Dialect Definition**: `include/ocamlc2/Dialect/OcamlDialect.td` defines MLIR types/ops
- **Tests**: `test/typecheck/` for type inference, `test/cg/` for code generation

## Development Notes

- The unifier implements Algorithm W with extensions for OCaml features like modules and variants
- MLIR operations include closure creation, environment capture, pattern matching, and OCaml runtime calls
- Module system supports first-class modules, functors with type constraints
- Tests use FileCheck for MLIR output verification
