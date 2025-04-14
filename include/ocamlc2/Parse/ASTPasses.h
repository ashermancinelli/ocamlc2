#pragma once
#include "ocamlc2/Parse/AST.h"

namespace ocamlc2 {

struct ASTPass {
  virtual void run(CompilationUnitAST *node) = 0;
  virtual ~ASTPass() = default;
private:
  void run(ASTNode *node);
};

struct ASTPassManager {
  void addPass(std::unique_ptr<ASTPass> pass);
  void addDefaultPasses();
  void run(CompilationUnitAST *node) const;
private:
  std::vector<std::unique_ptr<ASTPass>> passes;
};

}

