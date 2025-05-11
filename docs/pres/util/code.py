from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
import subprocess as sp

CODEFONT = "JetBrainsMono Nerd Font"

JBM = lambda *a, **kwa: Text(*a, **kwa, font=CODEFONT)
JBMI = lambda *a, **kw: JBM(*a, **kw, slant=ITALIC)

def SH(cmd: str) -> list[str]:
    return sp.check_output(cmd.split(' '), text=True).strip().split('\n')

def JBMCode(*a, **kw):
    print(a, kw)
    d = dict(line_spacing=1, background="window")
    kw = {**d, **kw}
    c = Code(*a, **kw)
    c.code = remove_invisible_chars(c.code)
    return c

def hl_code_line(code: Code, line: int):
    return (
        SurroundingRectangle(code.code[line])
        .set_fill(YELLOW)
        .set_opacity(0)
        .stretch_to_fit_width(code.background_mobject.width)
        .align_to(code.background_mobject, LEFT)
    )
