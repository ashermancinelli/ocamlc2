#pragma once
#include <string_view>

namespace ANSIColors {
  [[maybe_unused]] static constexpr std::string_view red = "\033[31m";
  [[maybe_unused]] static constexpr std::string_view green = "\033[32m";
  [[maybe_unused]] static constexpr std::string_view yellow = "\033[33m";
  [[maybe_unused]] static constexpr std::string_view blue = "\033[34m";
  [[maybe_unused]] static constexpr std::string_view reset = "\033[0m";
  [[maybe_unused]] static constexpr std::string_view bold = "\033[1m";
  [[maybe_unused]] static constexpr std::string_view italic = "\033[3m";
  [[maybe_unused]] static constexpr std::string_view underline = "\033[4m";
  [[maybe_unused]] static constexpr std::string_view reverse = "\033[7m";
  [[maybe_unused]] static constexpr std::string_view strikethrough = "\033[9m";
}
