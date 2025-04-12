#pragma once
#include <llvm/ADT/StringRef.h>

namespace ANSIColors {
  const char* red();
  const char* green();
  const char* yellow();
  const char* blue();
  const char* magenta();
  const char* cyan();
  const char* reset();
  const char* bold();
  const char* faint();
  const char* italic();
  const char* underline();
  const char* reverse();
  const char* strikethrough();
}
