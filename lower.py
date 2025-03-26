from xdsl.context import Context
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects import (
    affine,
    arith,
    func,
    memref,
    printf,
    scf,
)


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    return ctx


def compile(code: str) -> ModuleOp:
    context()
    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.blocks[0]))
    builder.insert(arith.ConstantOp.from_int_and_width(0, 32))
    return module
