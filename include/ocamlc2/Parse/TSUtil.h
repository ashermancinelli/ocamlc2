#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
inline ts::Node getNullNode() {
  return ts::Node{TSNode{{0, 0, 0, 0}, nullptr, nullptr}};
}
struct NamedCursor : public ts::Cursor {
  using ts::Cursor::Cursor;
  using value_type = ts::Node;
  NamedCursor(ts::Node node) : ts::Cursor(node.getCursor()) {}

  [[nodiscard]] bool operator++() {
     return gotoNextNamedSibling();
  }

  [[nodiscard]] value_type operator*() const {
    return getCurrentNode();
  }

  [[nodiscard]] bool gotoNextNamedSibling() {
    if (not gotoNextSibling())
      return false;
    return gotoNamedSibling();
  }

  [[nodiscard]] bool gotoFirstNamedChild() {
    if (not gotoFirstChild())
      return false;
    return gotoNamedSibling();
  }

  [[nodiscard]] bool isNamed() const { return getCurrentNode().isNamed(); }

private:
  [[nodiscard]] bool gotoNamedSibling() {
    while (not isNamed())
      if (not gotoNextSibling())
        return false;
    return true;
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ts::Point point);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ts::Extent<ts::Point> extent);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ts::Cursor cursor);
llvm::raw_ostream &
dump(llvm::raw_ostream &os, ts::Cursor cursor, std::string_view source,
     unsigned indent = 0, bool showUnnamed = false,
     std::optional<std::function<void(llvm::raw_ostream &, ts::Node)>> dumpNode=std::nullopt);
std::string_view getText(const ts::Node &node, std::string_view source);
llvm::SmallVector<ts::Node> getChildren(const ts::Node node);
llvm::SmallVector<ts::Node> getNamedChildren(const ts::Node node, llvm::ArrayRef<llvm::StringRef> ofTypes={});
llvm::SmallVector<ts::Node> getArguments(const ts::Node node);
bool isLetBindingRecursive(ts::Cursor ast);
inline std::optional<ts::Node> toOptional(ts::Node node) {
  return node.isNull() ? std::nullopt : std::optional{node};
}

struct ParameterDescriptor {
  enum LabelKind {
    None,
    Optional,
    Labeled,
  } labelKind;
  ParameterDescriptor() : labelKind(LabelKind::None) {}
  ParameterDescriptor(LabelKind labelKind, std::optional<llvm::StringRef> label,
                      std::optional<ts::Node> type,
                      std::optional<ts::Node> defaultValue)
      : labelKind(labelKind), label(label), type(type),
        defaultValue(defaultValue) {}
  std::optional<llvm::StringRef> label;
  std::optional<ts::Node> type;
  std::optional<ts::Node> defaultValue;
  [[nodiscard]] inline bool isNamed() const {
    return labelKind == LabelKind::Labeled or labelKind == LabelKind::Optional;
  }
  [[nodiscard]] inline bool isPositional() const { return not isNamed(); }
  [[nodiscard]] inline bool isOptional() const { return labelKind == LabelKind::Optional; }
  [[nodiscard]] inline bool hasDefaultValue() const { return defaultValue.has_value(); }
  template <typename T>
  friend T &operator<<(T &os, const ParameterDescriptor &desc) {
    os << "ParameterDescriptor(";
    os << "labelKind: " << desc.labelKind;
    os << ", label: " << desc.label;
    if (desc.type) {
      os << ", type: " << desc.type.value().getType();
    }
    if (desc.defaultValue) {
      os << ", defaultValue: " << desc.defaultValue.value().getType();
    }
    os << ")";
    return os;
  }
};

} // namespace ocamlc2

namespace llvm {
template<> struct DenseMapInfo<ts::Node> {
  static inline ts::Node getEmptyKey() { return ocamlc2::getNullNode(); }
  static inline ts::Node getTombstoneKey() { return ocamlc2::getNullNode(); }
  static unsigned getHashValue(const ts::Node &Val) { return Val.getID(); }
  static bool isEqual(const ts::Node &LHS, const ts::Node &RHS) { return LHS.getID() == RHS.getID(); }
};
}

namespace std {
template<> struct less<ts::Node> {
  bool operator()(const ts::Node &lhs, const ts::Node &rhs) const {
    return lhs.getID() < rhs.getID();
  }
};
}
