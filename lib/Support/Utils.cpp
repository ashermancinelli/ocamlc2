#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <llvm/Support/raw_ostream.h>
#define DEBUG_TYPE "utils"
#include "ocamlc2/Support/Debug.h.inc"
namespace fs = std::filesystem;

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

fs::path moduleNameToFilePath(std::string_view name) {
  auto path = std::string(name) + ".ml";
  std::transform(path.begin(), path.end(), path.begin(), ::tolower);
  return path;
}

std::string filePathToModuleName(fs::path path) {
  path = path.filename().replace_extension();
  auto newPathString = path.string();
  assert(newPathString.size() > 0);
  assert(newPathString.find('-') == std::string::npos);
  std::transform(newPathString.begin(), newPathString.end(), newPathString.begin(), ::tolower);
  newPathString[0] = std::toupper(newPathString[0]);
  return newPathString;
}

namespace ANSIColors {
using namespace ocamlc2::CL;
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

static constexpr std::string_view OCamlStdlibInterfaces[] = {
  "stdlib",
  "bool",
  "int",
  "string",
  "array",
  "list",
  "map",
  "option",
  "float",
  // "bytes.mli",
  // "char.mli",
  // "exn.mli",
  // "format.mli",
  // "set.mli",
};

static fs::path distDirFromExe(fs::path exe) {
  exe = fs::absolute(exe);
  auto bindir = exe.parent_path();
  auto install_root = bindir.parent_path();
  return install_root / "include" / "dist";
}

static fs::path resolvePath(fs::path exe, std::string_view interface) {
  static auto include_dir = distDirFromExe(exe);
  return include_dir / (std::string(interface) + ".mli");
}

SmallVector<fs::path> getStdlibOCamlInterfaceFiles(fs::path exe) {
  DBGS("exe: " << exe << "\n");
  if (ocamlc2::CL::StdlibOnly) {
    return {resolvePath(exe, "stdlib")};
  }
  return llvm::to_vector(llvm::map_range(OCamlStdlibInterfaces, [&](std::string_view interface) {
    return resolvePath(exe, interface);
  }));
}
