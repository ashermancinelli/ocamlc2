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
    d = dict(background="window")
    kw = {**d, **kw}
    c = Code(*a, **kw)
    return c

def hl_code_line(code: Code, line: int):
    return (
        SurroundingRectangle(code.code[line])
        .set_fill(YELLOW)
        .set_opacity(0)
        .stretch_to_fit_width(code.background_mobject.width)
        .align_to(code.background_mobject, LEFT)
    )


class HMIntro(Scene):
    """Introductory scene: Python vs C vs OCaml, IYSWIM & Hindley-Milner."""

    def construct(self):
        # ---------- 0. Title ----------
        title = Text("Type inference – If You See What I Mean", font_size=48)
        byline = Text("Landin → Hindley–Milner → OCaml", font_size=28)
        byline.next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(byline, shift=UP))
        self.wait(1)
        self.play(FadeOut(VGroup(title, byline)))

        # ---------- 1. Python vs OCaml one-liners ----------
        py_code = JBMCode(code_string="""def add(x, y):\n    return x + y""", language="python", background="window").scale(0.5)
        ml_code = JBMCode(code_string="""let add x y = x + y;;""", language="ocaml", background="window").scale(0.5)
        codes = VGroup(py_code, ml_code).arrange(RIGHT, buff=1)
        self.play(FadeIn(codes, shift=UP))
        self.wait(2)

        comment = Text("No annotations, but … runtime vs compile-time", font_size=28).next_to(codes, DOWN)
        self.play(Write(comment))
        self.wait(2)
        self.play(FadeOut(comment))

        # Highlight OCaml inference result
        inferred = Text("val add : int -> int -> int", font="Monospace", font_size=32, color=YELLOW)
        inferred.next_to(ml_code, DOWN)
        self.play(Write(inferred))
        self.wait(1)
        self.play(FadeOut(inferred))

        # ---------- 2. C snippet ----------
        self.play(codes.animate.shift(3 * UP).scale(0.8))
        c_code = JBMCode(code_string="""int add(int x, int y) {\n  return x + y;\n}\nvoid* v = (void*)add;""", language="c", background="window").scale(0.45)
        c_code.next_to(codes, DOWN, buff=1)
        self.play(FadeIn(c_code))
        self.wait(2)

        # Caption
        cap = Text("Explicit types – or escape hatch with void*", font_size=28)
        cap.next_to(c_code, DOWN)
        self.play(Write(cap))
        self.wait(2)

        # ---------- 3. The pitch ----------
        pitch_lines = [
            "Want Python-level expressivity", 
            "with C-level guarantees", 
            "…without writing the types", 
            "Compiler solves Sudoku (HM inference) – IYSWIM"]
        pitch = VGroup(*[Text(t, font_size=30) for t in pitch_lines]).arrange(DOWN, aligned_edge=LEFT)
        pitch.next_to(cap, DOWN, buff=1)
        self.play(FadeIn(pitch, shift=UP))
        self.wait(3)

        # ---------- 4. Hindley–Milner quick demo ----------
        self.play(FadeOut(VGroup(pitch, cap, c_code, codes)))

        # Reuse let-x-f-y animation inline (static for intro)
        code_line = Text("let x f y = f y;;", font="Monospace", font_size=40)
        self.play(Write(code_line))
        self.wait(0.5)

        explain = Text("Compiler introduces 'a, 'b, 'c … and unifies", font_size=28)
        explain.next_to(code_line, DOWN)
        self.play(FadeIn(explain, shift=UP))
        self.wait(2)

        result = Text("⇒ ('c → 'd) → 'c → 'd", font="Monospace", font_size=36, color=GREEN)
        result.next_to(explain, DOWN)
        self.play(Write(result))
        self.wait(2)

        # ---------- 5. Landin reference ----------
        self.play(FadeOut(VGroup(explain)))
        landin = Text("Peter Landin, 'The Next 700 Programming Languages' (1966)", font_size=24)
        landin.next_to(result, DOWN, buff=1)
        self.play(FadeIn(landin))
        self.wait(3)

        # ---------- 6. Outro ----------
        outro = Text("Full static types, zero annotations – If-You-See-What-I-Mean!", font_size=32)
        self.play(FadeOut(VGroup(code_line, result, landin)))
        self.play(Write(outro))
        self.wait(2)

        self.play(FadeOut(outro))


if __name__ == "__main__":
    from manim import config
    config.pixel_height = 720
    config.pixel_width = 1280
    HMIntro().render() 
