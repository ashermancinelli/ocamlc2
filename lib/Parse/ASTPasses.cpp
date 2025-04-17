#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/Unifier.h"
#define DEBUG_TYPE "ocamlc2-ast-passes"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

class TupleFixupPass : public ASTPass {
  void run(CompilationUnitAST *node) override {
  }
};

void ASTPassManager::addPass(std::unique_ptr<ASTPass> pass) {
  passes.push_back(std::move(pass));
}

void ASTPass::runOnNode(ASTNode *node) {
}

void ASTPassManager::addDefaultPasses() {
  addPass(std::make_unique<TypeCheckingPass>());
}

void ASTPassManager::run(CompilationUnitAST *node) const {
  for (auto &pass : passes) {
    pass->run(node);
  }
}

}
