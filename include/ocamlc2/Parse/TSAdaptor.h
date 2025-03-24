#pragma once
#include <string>
#include "ocamlc2/Support/LLVMCommon.h"

struct TSTree;
FailureOr<TSTree *> parseOCaml(const std::string &source);
FailureOr<std::string> slurpFile(const std::string &path);
