#include "ocamlc2/CamlParse/Parse.h"
#include "ocamlc2/CamlParse/AST.h"
#include "ocamlc2/CamlParse/Lex.h"
#include "llvm/Support/raw_ostream.h"

namespace ocamlc2 {
inline namespace CamlParse {

// Helper function to create proper indentation
static void indent(llvm::raw_ostream& OS, unsigned Level) {
  for (unsigned i = 0; i < Level; ++i)
    OS << "  ";
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, Token tok) {
  return OS << getTokenName(tok);
}

// Forward declaration of dump function
static void dumpASTNode(llvm::raw_ostream& OS, const ASTNode* Node, unsigned IndentLevel);

// Dump a list of AST nodes with proper indentation
template <typename T>
static void dumpNodes(llvm::raw_ostream& OS, const std::vector<std::unique_ptr<T>>& Nodes, 
                    unsigned IndentLevel, const char* Label = nullptr) {
  if (Label)
    OS << Label << ":\n";
  
  for (const auto& Node : Nodes) {
    dumpASTNode(OS, Node.get(), IndentLevel);
  }
}

// Dump a location
static void dumpLocation(llvm::raw_ostream& OS, const Location& Loc) {
  OS << "[" << Loc.getStartLine() << ":" << Loc.getStartCol() 
     << "-" << Loc.getEndLine() << ":" << Loc.getEndCol() << "]";
}

// Dump a constant
static void dumpConstant(llvm::raw_ostream& OS, const ConstantAST* Node, unsigned IndentLevel) {
  indent(OS, IndentLevel);
  
  switch (Node->getConstantKind()) {
    case ConstantAST::Const_Int: {
      auto* IntNode = static_cast<const IntConstantAST*>(Node);
      OS << "IntConstant: " << IntNode->getValue();
      if (IntNode->getSuffix())
        OS << " (suffix: " << *IntNode->getSuffix() << ")";
      OS << "\n";
      break;
    }
    case ConstantAST::Const_Char: {
      auto* CharNode = static_cast<const CharConstantAST*>(Node);
      OS << "CharConstant: '" << CharNode->getValue() << "'\n";
      break;
    }
    case ConstantAST::Const_Float: {
      auto* FloatNode = static_cast<const FloatConstantAST*>(Node);
      OS << "FloatConstant: " << FloatNode->getValue();
      if (FloatNode->getSuffix())
        OS << " (suffix: " << *FloatNode->getSuffix() << ")";
      OS << "\n";
      break;
    }
    case ConstantAST::Const_String: {
      auto* StringNode = static_cast<const StringConstantAST*>(Node);
      OS << "StringConstant: \"" << StringNode->getValue() << "\"";
      if (StringNode->getDelimiter())
        OS << " (delimiter: " << *StringNode->getDelimiter() << ")";
      OS << "\n";
      break;
    }
    case ConstantAST::Const_Int32: {
      auto* Int32Node = static_cast<const Int32ConstantAST*>(Node);
      OS << "Int32Constant: " << Int32Node->getValue() << "\n";
      break;
    }
    case ConstantAST::Const_Int64: {
      auto* Int64Node = static_cast<const Int64ConstantAST*>(Node);
      OS << "Int64Constant: " << Int64Node->getValue() << "\n";
      break;
    }
    case ConstantAST::Const_Nativeint: {
      auto* NativeintNode = static_cast<const NativeintConstantAST*>(Node);
      OS << "NativeintConstant: " << NativeintNode->getValue() << "\n";
      break;
    }
  }
}

// Helper to convert RecFlag to string
static const char* toString(RecFlag Flag) {
  return Flag == RecFlag::Recursive ? "Recursive" : "Nonrecursive";
}

// Helper to convert ArgLabel to string
static const char* toString(ArgLabel Label) {
  switch (Label) {
    case ArgLabel::Nolabel: return "Nolabel";
    case ArgLabel::Labelled: return "Labelled";
    case ArgLabel::Optional: return "Optional";
  }
}

// Helper to convert DirectionFlag to string
static const char* toString(DirectionFlag Dir) {
  return Dir == DirectionFlag::Upto ? "Upto" : "Downto";
}

// Helper to convert PrivateFlag to string
static const char* toString(PrivateFlag Flag) {
  return Flag == PrivateFlag::Private ? "Private" : "Public";
}

// Helper to convert MutableFlag to string
static const char* toString(MutableFlag Flag) {
  return Flag == MutableFlag::Mutable ? "Mutable" : "Immutable";
}

// Helper to convert Variance to string
static const char* toString(Variance V) {
  switch (V) {
    case Variance::Covariant: return "Covariant";
    case Variance::Contravariant: return "Contravariant";
    case Variance::NoVariance: return "NoVariance";
    case Variance::Bivariant: return "Bivariant";
  }
}

// Helper to convert Injectivity to string
static const char* toString(Injectivity I) {
  return I == Injectivity::Injective ? "Injective" : "NoInjectivity";
}

// Main function to dump AST nodes
static void dumpASTNode(llvm::raw_ostream& OS, const ASTNode* Node, unsigned IndentLevel) {
  if (!Node) {
    indent(OS, IndentLevel);
    OS << "<null>\n";
    return;
  }

  switch (Node->getKind()) {
    // Compilation unit
    case ASTNode::Node_Compilation_Unit: {
      auto* CU = static_cast<const CompilationUnitAST*>(Node);
      indent(OS, IndentLevel);
      OS << "CompilationUnit ";
      dumpLocation(OS, CU->getLoc());
      OS << "\n";
      dumpNodes(OS, CU->getStructures(), IndentLevel + 1, "Structures");
      break;
    }
    
    // Structure
    case ASTNode::Node_Structure: {
      auto* Struct = static_cast<const StructureAST*>(Node);
      indent(OS, IndentLevel);
      OS << "Structure ";
      dumpLocation(OS, Struct->getLoc());
      OS << "\n";
      dumpNodes(OS, Struct->getItems(), IndentLevel + 1, "Items");
      break;
    }
    
    // Structure item
    case ASTNode::Node_Structure_Item: {
      auto* Item = static_cast<const StructureItemAST*>(Node);
      indent(OS, IndentLevel);
      OS << "StructureItem (Kind: ";
      
      switch (Item->getStructureItemKind()) {
        case StructureItemAST::Str_Value:
          OS << "Value";
          if (const auto* ValueItem = llvm::dyn_cast<StructureValueAST>(Item)) {
            OS << ", " << toString(ValueItem->getRecursiveFlag());
            OS << ") ";
            dumpLocation(OS, Item->getLoc());
            OS << "\n";
            dumpNodes(OS, ValueItem->getDefinitions(), IndentLevel + 1, "Definitions");
          } else {
            OS << ") ";
            dumpLocation(OS, Item->getLoc());
            OS << "\n";
          }
          break;
          
        case StructureItemAST::Str_Type:
          OS << "Type) ";
          dumpLocation(OS, Item->getLoc());
          OS << "\n";
          if (const auto* TypeItem = llvm::dyn_cast<StructureTypeAST>(Item)) {
            dumpNodes(OS, TypeItem->getDeclarations(), IndentLevel + 1, "Declarations");
          }
          break;
          
        default:
          OS << "Unknown) ";
          dumpLocation(OS, Item->getLoc());
          OS << "\n";
          break;
      }
      break;
    }
    
    // Value definition
    case ASTNode::Node_Value_Definition: {
      auto* Def = static_cast<const ValueDefinitionAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ValueDefinition (" << toString(Def->getRecursiveFlag()) << ") ";
      dumpLocation(OS, Def->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Pattern:\n";
      dumpASTNode(OS, Def->getPattern(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Expression:\n";
      dumpASTNode(OS, Def->getExpression(), IndentLevel + 2);
      break;
    }
    
    // Type declaration
    case ASTNode::Node_Type_Declaration: {
      auto* TypeDecl = static_cast<const TypeDeclarationAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypeDeclaration (" << toString(TypeDecl->getRecursiveFlag())
         << ", " << toString(TypeDecl->getPrivateFlag()) << ") '"
         << TypeDecl->getName() << "' ";
      dumpLocation(OS, TypeDecl->getLoc());
      OS << "\n";
      
      // Parameters
      const auto& Params = TypeDecl->getParameters();
      if (!Params.empty()) {
        indent(OS, IndentLevel + 1);
        OS << "Parameters:\n";
        for (const auto& P : Params) {
          indent(OS, IndentLevel + 2);
          OS << "'" << P.getName() << "' (" 
             << toString(P.getVariance()) << ", "
             << toString(P.getInjectivity()) << ")\n";
        }
      }
      
      // Constructors
      const auto& Constructors = TypeDecl->getConstructors();
      if (!Constructors.empty()) {
        indent(OS, IndentLevel + 1);
        OS << "Constructors:\n";
        for (const auto& C : Constructors) {
          indent(OS, IndentLevel + 2);
          OS << "'" << C.getName() << "' (" 
             << toString(C.getMutableFlag()) << ", "
             << toString(C.getPrivateFlag()) << ")\n";
        }
      }
      
      // Manifest type
      if (TypeDecl->getManifest()) {
        indent(OS, IndentLevel + 1);
        OS << "Manifest Type:\n";
        dumpASTNode(OS, TypeDecl->getManifest()->get(), IndentLevel + 2);
      }
      break;
    }
    
    // Patterns
    case ASTNode::Node_Pattern_Variable: {
      auto* Var = static_cast<const PatternVariableAST*>(Node);
      indent(OS, IndentLevel);
      OS << "PatternVariable: '" << Var->getName() << "' ";
      dumpLocation(OS, Var->getLoc());
      OS << "\n";
      break;
    }
    
    case ASTNode::Node_Pattern_Constant: {
      auto* Const = static_cast<const PatternConstantAST*>(Node);
      indent(OS, IndentLevel);
      OS << "PatternConstant ";
      dumpLocation(OS, Const->getLoc());
      OS << "\n";
      dumpConstant(OS, Const->getConstant(), IndentLevel + 1);
      break;
    }
    
    case ASTNode::Node_Pattern_Tuple: {
      auto* Tuple = static_cast<const PatternTupleAST*>(Node);
      indent(OS, IndentLevel);
      OS << "PatternTuple ";
      dumpLocation(OS, Tuple->getLoc());
      OS << "\n";
      dumpNodes(OS, Tuple->getElements(), IndentLevel + 1, "Elements");
      break;
    }
    
    case ASTNode::Node_Pattern_Construct: {
      auto* Construct = static_cast<const PatternConstructAST*>(Node);
      indent(OS, IndentLevel);
      OS << "PatternConstruct: '" << Construct->getConstructor() << "' ";
      dumpLocation(OS, Construct->getLoc());
      OS << "\n";
      
      if (Construct->getArgument()) {
        indent(OS, IndentLevel + 1);
        OS << "Argument:\n";
        dumpASTNode(OS, Construct->getArgument()->get(), IndentLevel + 2);
      }
      break;
    }
    
    case ASTNode::Node_Pattern_Any: {
      auto* Any = static_cast<const PatternAnyAST*>(Node);
      indent(OS, IndentLevel);
      OS << "PatternAny ";
      dumpLocation(OS, Any->getLoc());
      OS << "\n";
      break;
    }
    
    // Expressions
    case ASTNode::Node_Expression_Constant: {
      auto* Const = static_cast<const ExpressionConstantAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionConstant ";
      dumpLocation(OS, Const->getLoc());
      OS << "\n";
      dumpConstant(OS, Const->getConstant(), IndentLevel + 1);
      break;
    }
    
    case ASTNode::Node_Expression_Variable: {
      auto* Var = static_cast<const ExpressionVariableAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionVariable: '" << Var->getName() << "' ";
      dumpLocation(OS, Var->getLoc());
      OS << "\n";
      break;
    }
    
    case ASTNode::Node_Expression_Let: {
      auto* Let = static_cast<const ExpressionLetAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionLet (" << toString(Let->getRecursiveFlag()) << ") ";
      dumpLocation(OS, Let->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Let->getDefinitions(), IndentLevel + 1, "Definitions");
      
      indent(OS, IndentLevel + 1);
      OS << "Body:\n";
      dumpASTNode(OS, Let->getBody(), IndentLevel + 2);
      break;
    }
    
    case ASTNode::Node_Expression_Function: {
      auto* Func = static_cast<const ExpressionFunctionAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionFunction ";
      dumpLocation(OS, Func->getLoc());
      OS << "\n";
      
      // Parameters
      indent(OS, IndentLevel + 1);
      OS << "Parameters:\n";
      for (const auto& Param : Func->getParameters()) {
        indent(OS, IndentLevel + 2);
        OS << toString(Param->getLabel());
        if (!Param->getLabelName().empty())
          OS << " '" << Param->getLabelName() << "'";
        OS << ":\n";
        dumpASTNode(OS, Param->getPattern(), IndentLevel + 3);
      }
      
      // Return type
      if (Func->getReturnType()) {
        indent(OS, IndentLevel + 1);
        OS << "ReturnType:\n";
        dumpASTNode(OS, Func->getReturnType()->get(), IndentLevel + 2);
      }
      
      // Body
      indent(OS, IndentLevel + 1);
      OS << "Body:\n";
      dumpASTNode(OS, Func->getBody(), IndentLevel + 2);
      break;
    }
    
    case ASTNode::Node_Expression_Apply: {
      auto* Apply = static_cast<const ExpressionApplyAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionApply ";
      dumpLocation(OS, Apply->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Function:\n";
      dumpASTNode(OS, Apply->getFunction(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Arguments:\n";
      for (const auto& Arg : Apply->getArguments()) {
        indent(OS, IndentLevel + 2);
        OS << toString(Arg.first) << ":\n";
        dumpASTNode(OS, Arg.second.get(), IndentLevel + 3);
      }
      break;
    }
    
    case ASTNode::Node_Expression_Match: {
      auto* Match = static_cast<const ExpressionMatchAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionMatch ";
      dumpLocation(OS, Match->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Expression:\n";
      dumpASTNode(OS, Match->getExpression(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Cases:\n";
      for (const auto& Case : Match->getCases()) {
        indent(OS, IndentLevel + 2);
        OS << "Case:\n";
        
        indent(OS, IndentLevel + 3);
        OS << "Pattern:\n";
        dumpASTNode(OS, Case->getPattern(), IndentLevel + 4);
        
        if (Case->getGuard()) {
          indent(OS, IndentLevel + 3);
          OS << "Guard:\n";
          dumpASTNode(OS, Case->getGuard()->get(), IndentLevel + 4);
        }
        
        indent(OS, IndentLevel + 3);
        OS << "Expression:\n";
        dumpASTNode(OS, Case->getExpression(), IndentLevel + 4);
      }
      break;
    }
    
    case ASTNode::Node_Expression_Ifthenelse: {
      auto* If = static_cast<const ExpressionIfthenelseAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionIfthenelse ";
      dumpLocation(OS, If->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Condition:\n";
      dumpASTNode(OS, If->getCondition(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Then:\n";
      dumpASTNode(OS, If->getThenExpr(), IndentLevel + 2);
      
      if (If->getElseExpr()) {
        indent(OS, IndentLevel + 1);
        OS << "Else:\n";
        dumpASTNode(OS, If->getElseExpr()->get(), IndentLevel + 2);
      }
      break;
    }
    
    case ASTNode::Node_Expression_For: {
      auto* For = static_cast<const ExpressionForAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionFor (" << toString(For->getDirection()) << ") ";
      dumpLocation(OS, For->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Pattern:\n";
      dumpASTNode(OS, For->getPattern(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Start:\n";
      dumpASTNode(OS, For->getStartExpr(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "End:\n";
      dumpASTNode(OS, For->getEndExpr(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Body:\n";
      dumpASTNode(OS, For->getBody(), IndentLevel + 2);
      break;
    }
    
    case ASTNode::Node_Expression_While: {
      auto* While = static_cast<const ExpressionWhileAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionWhile ";
      dumpLocation(OS, While->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Condition:\n";
      dumpASTNode(OS, While->getCondition(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Body:\n";
      dumpASTNode(OS, While->getBody(), IndentLevel + 2);
      break;
    }
    
    case ASTNode::Node_Expression_Sequence: {
      auto* Seq = static_cast<const ExpressionSequenceAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionSequence ";
      dumpLocation(OS, Seq->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Seq->getExpressions(), IndentLevel + 1, "Expressions");
      break;
    }
    
    case ASTNode::Node_Expression_Construct: {
      auto* Construct = static_cast<const ExpressionConstructAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionConstruct: '" << Construct->getConstructor() << "' ";
      dumpLocation(OS, Construct->getLoc());
      OS << "\n";
      
      if (Construct->getArgument()) {
        indent(OS, IndentLevel + 1);
        OS << "Argument:\n";
        dumpASTNode(OS, Construct->getArgument()->get(), IndentLevel + 2);
      }
      break;
    }
    
    case ASTNode::Node_Expression_Tuple: {
      auto* Tuple = static_cast<const ExpressionTupleAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionTuple ";
      dumpLocation(OS, Tuple->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Tuple->getElements(), IndentLevel + 1, "Elements");
      break;
    }
    
    case ASTNode::Node_Expression_Array: {
      auto* Array = static_cast<const ExpressionArrayAST*>(Node);
      indent(OS, IndentLevel);
      OS << "ExpressionArray ";
      dumpLocation(OS, Array->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Array->getElements(), IndentLevel + 1, "Elements");
      break;
    }
    
    // Types
    case ASTNode::Node_Type_Var: {
      auto* Var = static_cast<const TypeVarAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypeVar: '" << Var->getName() << "' ";
      dumpLocation(OS, Var->getLoc());
      OS << "\n";
      break;
    }
    
    case ASTNode::Node_Type_Constr: {
      auto* Constr = static_cast<const TypeConstrAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypeConstr: '" << Constr->getName() << "' ";
      dumpLocation(OS, Constr->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Constr->getArguments(), IndentLevel + 1, "Arguments");
      break;
    }
    
    case ASTNode::Node_Type_Arrow: {
      auto* Arrow = static_cast<const TypeArrowAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypeArrow ";
      dumpLocation(OS, Arrow->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Left:\n";
      dumpASTNode(OS, Arrow->getLeft(), IndentLevel + 2);
      
      indent(OS, IndentLevel + 1);
      OS << "Right:\n";
      dumpASTNode(OS, Arrow->getRight(), IndentLevel + 2);
      break;
    }
    
    case ASTNode::Node_Type_Tuple: {
      auto* Tuple = static_cast<const TypeTupleAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypeTuple ";
      dumpLocation(OS, Tuple->getLoc());
      OS << "\n";
      
      dumpNodes(OS, Tuple->getElements(), IndentLevel + 1, "Elements");
      break;
    }
    
    case ASTNode::Node_Type_Poly: {
      auto* Poly = static_cast<const TypePolyAST*>(Node);
      indent(OS, IndentLevel);
      OS << "TypePoly ";
      dumpLocation(OS, Poly->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Variables: ";
      bool First = true;
      for (const auto& Var : Poly->getVariables()) {
        if (!First) OS << ", ";
        OS << "'" << Var;
        First = false;
      }
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Type:\n";
      dumpASTNode(OS, Poly->getType(), IndentLevel + 2);
      break;
    }
    
    // Parameters
    case ASTNode::Node_Parameter: {
      auto* Param = static_cast<const ParameterAST*>(Node);
      indent(OS, IndentLevel);
      OS << "Parameter (" << toString(Param->getLabel());
      if (!Param->getLabelName().empty())
        OS << " '" << Param->getLabelName() << "'";
      OS << ") ";
      dumpLocation(OS, Param->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Pattern:\n";
      dumpASTNode(OS, Param->getPattern(), IndentLevel + 2);
      
      if (Param->getDefaultValue()) {
        indent(OS, IndentLevel + 1);
        OS << "DefaultValue:\n";
        dumpASTNode(OS, Param->getDefaultValue()->get(), IndentLevel + 2);
      }
      break;
    }
    
    // Match cases
    case ASTNode::Node_Match_Case: {
      auto* Case = static_cast<const MatchCaseAST*>(Node);
      indent(OS, IndentLevel);
      OS << "MatchCase ";
      dumpLocation(OS, Case->getLoc());
      OS << "\n";
      
      indent(OS, IndentLevel + 1);
      OS << "Pattern:\n";
      dumpASTNode(OS, Case->getPattern(), IndentLevel + 2);
      
      if (Case->getGuard()) {
        indent(OS, IndentLevel + 1);
        OS << "Guard:\n";
        dumpASTNode(OS, Case->getGuard()->get(), IndentLevel + 2);
      }
      
      indent(OS, IndentLevel + 1);
      OS << "Expression:\n";
      dumpASTNode(OS, Case->getExpression(), IndentLevel + 2);
      break;
    }
    
    default:
      indent(OS, IndentLevel);
      OS << "Unknown Node (Kind: " << Node->getKind() << ") ";
      dumpLocation(OS, Node->getLoc());
      OS << "\n";
      break;
  }
}

// Implement the operator<< for ASTNode
llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const ASTNode& Node) {
  dumpASTNode(OS, &Node, 0);
  return OS;
}

} // inline namespace CamlParse
} // namespace ocamlc2
