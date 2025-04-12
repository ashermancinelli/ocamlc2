#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <fstream>
#include <iostream>

FailureOr<std::string> slurpFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    exit(1);
  }
  
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  return contents;
}

namespace ANSIColors {
const char* red() { return Color ? "\033[31m" : ""; }
const char* green() { return Color ? "\033[32m" : ""; }
const char* yellow() { return Color ? "\033[33m" : ""; }
const char* blue() { return Color ? "\033[34m" : ""; }
const char* magenta() { return Color ? "\033[35m" : ""; }
const char* cyan() { return Color ? "\033[36m" : ""; }
const char* reset() { return Color ? "\033[0m" : ""; }
const char* bold() { return Color ? "\033[1m" : ""; }
const char* faint() { return Color ? "\033[2m" : ""; }
const char* italic() { return Color ? "\033[3m" : ""; }
const char* underline() { return Color ? "\033[4m" : ""; }
const char* reverse() { return Color ? "\033[7m" : ""; }
const char* strikethrough() { return Color ? "\033[9m" : ""; }
} // namespace ANSIColors
