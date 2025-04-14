#pragma once

#include <memory>
#include <string>
#include <istream>
#include "ocamlc2/CamlParse/AST.h"
#include "ocamlc2/Support/LLVMCommon.h"

namespace ocamlc2 {
inline namespace CamlParse {
std::unique_ptr<CompilationUnitAST> parse(std::istream &file, llvm::StringRef filepath);
} // inline namespace CamlParse
} // namespace ocamlc2
