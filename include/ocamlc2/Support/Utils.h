#pragma once
#include <string>
#include "mlir/IR/MLIRContext.h"

mlir::FailureOr<std::string> slurpFile(const std::string &path);
