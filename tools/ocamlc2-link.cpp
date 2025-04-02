#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdlib>
#include <system_error>
#include <optional>
#include <sstream>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "OcamlC2Config.h"

#define DEBUG_TYPE "ocamlc2-link"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;
namespace cl = llvm::cl;

template <typename T>
using FailureOr = llvm::Expected<T>;

static cl::opt<std::string> InputFile(cl::Positional, cl::desc("<input LLVM IR file>"), cl::Required);
static cl::opt<std::string> OutputFile("o", cl::desc("Output executable filename"), cl::value_desc("filename"));
static cl::list<std::string> LinkOptions("l", cl::desc("Additional libraries to link"));
static cl::list<std::string> ClangOptions("Xclang", cl::desc("Additional options to pass to clang"));
static cl::opt<bool> Verbose("v", cl::desc("Enable verbose output"));

// Add platform-specific flags from the config
void addPlatformFlags(std::vector<std::string>& args) {
  // Add common flags first
  const char* commonFlags = OCAMLC2_LINKER_FLAGS;
  if (commonFlags && commonFlags[0] != '\0') {
    std::istringstream iss(commonFlags);
    std::string flag;
    while (std::getline(iss, flag, ' ')) {
      if (!flag.empty()) {
        args.push_back(flag);
      }
    }
  }
}

FailureOr<std::string> getExecutablePath() {
  char buffer[1024];
  size_t size = sizeof(buffer);
  
  #if defined(_WIN32)
  GetModuleFileNameA(NULL, buffer, size);
  return std::string(buffer);
  #elif defined(__APPLE__)
  uint32_t bufsize = static_cast<uint32_t>(size);
  if (_NSGetExecutablePath(buffer, &bufsize) != 0) {
    return llvm::createStringError(std::errc::bad_address, "Failed to get executable path, buffer too small");
  }
  char realPath[PATH_MAX];
  if (realpath(buffer, realPath) == nullptr) {
    return llvm::createStringError(std::errc::bad_address, "Failed to resolve executable path");
  }
  return std::string(realPath);
  #else
  ssize_t len = readlink("/proc/self/exe", buffer, size - 1);
  if (len == -1) {
    return llvm::createStringError(std::errc::bad_address, "Failed to get executable path");
  }
  buffer[len] = '\0';
  return std::string(buffer);
  #endif
}

FailureOr<fs::path> findRuntimeLibrary() {
  auto exePathOrErr = getExecutablePath();
  if (!exePathOrErr)
    return exePathOrErr.takeError();
  
  std::string exePath = exePathOrErr.get();
  fs::path exeDir = fs::path(exePath).parent_path();
  fs::path libPath = exeDir / "../lib/libocamlrt_static.a";
  
  if (fs::exists(libPath)) {
    return fs::absolute(libPath);
  } else {
    return llvm::createStringError(std::errc::no_such_file_or_directory, 
                                  "Runtime library not found at %s", 
                                  libPath.string().c_str());
  }
}

FailureOr<std::string> findClang() {
  auto clangPath = llvm::sys::findProgramByName("clang");
  if (!clangPath) {
    return llvm::createStringError(std::errc::no_such_file_or_directory, "clang not found in PATH");
  }
  return clangPath.get();
}

llvm::Error linkWithClang(const std::string& clangPath, const fs::path& runtimeLib, const std::string& irFile) {
  std::vector<std::string> args;
  args.push_back(clangPath);
  args.push_back(irFile);
  args.push_back(runtimeLib.string());
  
  // Add output file
  if (!OutputFile.empty()) {
    args.push_back("-o");
    args.push_back(OutputFile);
  }
  
  // Add platform-specific flags from the config
  addPlatformFlags(args);
  
  // Add additional libraries
  for (const auto& lib : LinkOptions) {
    args.push_back("-l" + lib);
  }
  
  // Add additional clang options
  for (const auto& opt : ClangOptions) {
    args.push_back(opt);
  }
  
  // Print command in verbose mode
  if (Verbose) {
    llvm::outs() << "Executing: ";
    for (const auto& arg : args) {
      // Quote arguments containing spaces
      if (arg.find(' ') != std::string::npos)
        llvm::outs() << "\"" << arg << "\" ";
      else
        llvm::outs() << arg << " ";
    }
    llvm::outs() << "\n";
  }
  
  std::vector<llvm::StringRef> argsRef;
  for (const auto& arg : args) {
    argsRef.push_back(arg);
  }

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(clangPath, argsRef, std::nullopt, {}, 0, 0, &errMsg);
  
  if (result != 0) {
    return llvm::createStringError(std::errc::executable_format_error, 
                                  "Error linking: %s", errMsg.c_str());
  }
  
  return llvm::Error::success();
}

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "OCaml IR Linker\n");
  
  // Find runtime library
  auto runtimeLibOrErr = findRuntimeLibrary();
  if (!runtimeLibOrErr) {
    llvm::errs() << "Error: " << llvm::toString(runtimeLibOrErr.takeError()) << "\n";
    return 1;
  }
  fs::path runtimeLib = runtimeLibOrErr.get();
  DBGS("Found runtime library: " << runtimeLib << "\n");
  
  // Find clang
  auto clangPathOrErr = findClang();
  if (!clangPathOrErr) {
    llvm::errs() << "Error: " << llvm::toString(clangPathOrErr.takeError()) << "\n";
    return 1;
  }
  std::string clangPath = clangPathOrErr.get();
  DBGS("Found clang: " << clangPath << "\n");
  
  // Link the IR file with the runtime library
  auto err = linkWithClang(clangPath, runtimeLib, InputFile);
  if (err) {
    llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  
  if (Verbose)
    llvm::outs() << "Linking successful\n";
  return 0;
}
