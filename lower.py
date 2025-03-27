import logging
from typing import Optional

from tree_sitter import Node, Tree, TreeCursor
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import (affine, arith, builtin, func, llvm, memref, printf,
                           scf)
from xdsl.dialects.builtin import Builtin, IndexType, ModuleOp, i8, i32
from xdsl.dialects.llvm import AllocaOp
from xdsl.ir import BlockArgument, SSAValue
from xdsl.rewriter import InsertPoint

logger = logging.getLogger(__name__)


class Lower:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.ctx = Lower.context()
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))
        self.ptr_type = llvm.LLVMPointerType.typed(i8)
        self.ptrint_type = llvm.LLVMPointerType.typed(i32)
        self.unit: Optional[SSAValue] = None

    def one(self, width: int = 64) -> arith.ConstantOp:
        return arith.ConstantOp.from_int_and_width(1, width)

    def zero(self, width: int = 64) -> arith.ConstantOp:
        return arith.ConstantOp.from_int_and_width(0, width)

    def store(self, val: SSAValue, ptr: SSAValue) -> None:
        if val == ptr:
            return
        llvm.StoreOp(val, ptr)

    def lower_node(self, node: Node) -> Optional[SSAValue]:
        assert node is not None
        logger.debug(f"lowering node of type: {node.type}")
        match node.type:
            case "compilation_unit":
                ret = None
                for child in node.children:
                    ret = self.lower_node(child)
                assert ret
                return ret
            case "comment":
                pass
            case "value_definition":
                let, let_binding = node.children
                lhs = let_binding.child_by_field_name("pattern")
                rhs = let_binding.child_by_field_name("body")
                assert let and lhs and rhs
                lhs_val = self.lower_node(lhs)
                rhs_val = self.lower_node(rhs)
                assert lhs_val and rhs_val
                self.store(rhs_val, lhs_val)
                return lhs_val
            case "unit":
                if self.unit is None:
                    self.unit = AllocaOp(self.one(), i32).results[0]
                    llvm.StoreOp(self.zero(), self.unit)
                assert self.unit
                return self.unit
            case "for_expression":

                @Builder.implicit_region((IndexType(),))
                def body(_: tuple[BlockArgument, ...]) -> None:
                    scf.YieldOp()

                for_op = scf.ForOp(
                    self.zero(IndexType()),
                    self.one(IndexType()),
                    self.one(IndexType()),
                    [],
                    body,
                )
                assert for_op
                return self.unit
            case _:
                raise ValueError(f"unknown node type: {node.type}")

    def lower_cursor(self, cursor: TreeCursor) -> Optional[SSAValue]:
        assert cursor.node is not None
        logger.debug(f"lowering cursor of type: {cursor.node.type}")
        match cursor.node.type:
            case _:
                raise ValueError(f"unknown cursor type: {cursor.node.type}")
        return None

    def lower(self) -> ModuleOp:
        root = self.tree.root_node
        assert root.type == "compilation_unit"
        main = func.FuncOp("main", ((), (i32,)))
        with ImplicitBuilder(main.body):
            self.lower_node(root)
            ret = arith.ConstantOp.from_int_and_width(0, i32)
            func.ReturnOp(ret)

        self.builder.insert(main)
        self.module.verify()

        return self.module

    @staticmethod
    def context() -> Context:
        ctx = Context()
        ctx.load_dialect(affine.Affine)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(Builtin)
        ctx.load_dialect(llvm.LLVM)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(memref.MemRef)
        ctx.load_dialect(printf.Printf)
        ctx.load_dialect(scf.Scf)
        return ctx
