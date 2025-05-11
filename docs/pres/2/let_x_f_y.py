from manim import *
import sys
import os

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, "..", "..", "util"))
from code import *


class Code:
    def __init__(self, scene):
        self.scene = scene
        self.parts = {
            "let": "let",
            "x": "x",
            "f": "f",
            "y": "y",
            "eq": "=",
            "f_y": "f y",
        }

        self._text = ""
        for i, (k, v) in enumerate(self.parts.items()):
            if i != 0:
                self._text += " "
            start = len(self._text)
            self._text += v
            end = len(self._text)
            spaces = self._text.count(" ")
            r = (start - spaces, end - spaces)
            self.parts[k] = r

        self._text = Text(self._text, font="Monospace", font_size=40)

        self.indications = {
            k: Indicate(self._text[self.parts[k][0] : self.parts[k][1]])
            for k in self.parts
        }

    @property
    def text(self):
        return self._text

    def indicate(self, part):
        return self.indications[part]

    def highlight(self, part):
        start, end = self.parts[part]
        t = f"{start} {end}"
        t = Text(t, font="Monospace", font_size=40).shift(DOWN)
        self.scene.play(Write(t))
        self.scene.play(FadeOut(t))
        return Indicate(self.text[start:end])


class HindleyMilnerInference(Scene):
    def construct(self):
        code = Code(self)
        code.text.shift(UP)
        self.play(Write(code.text))

        # Helper to build a VGroup of Text lines left-aligned under the code.
        def build_env_block(lines: list[str]) -> VGroup:
            """Create a VGroup of left-aligned Text lines stacked vertically, then center the block under the code."""
            line_mobs = [Text(line, font="Monospace", font_size=28) for line in lines]

            # Arrange lines vertically with their left edges aligned.
            block = VGroup(*line_mobs).arrange(DOWN, aligned_edge=LEFT)

            # Position the block below the code.
            block.next_to(code.text, DOWN, buff=0.5)

            # Center the whole block horizontally with respect to the code.
            block.move_to(np.array([code.text.get_center()[0], block.get_center()[1], 0]))

            return block

        steps: list[tuple[str, list[str]]] = [
            ("let", ["(* Environment: *)"]),
            (
                "x",
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                ],
            ),
            (
                "f",
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                    "(* f: 'b (fresh) *)",
                ],
            ),
            (
                "y",
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                    "(* f: 'b (fresh) *)",
                    "(* y: 'c (fresh) *)",
                ],
            ),
            (
                "f_y",
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                    "(* f: 'b must be a function! *)",
                    "(* y: 'c (fresh) *)",
                ],
            ),
            (
                None,
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                    "(* f: 'c → 'd (refined) *)",
                    "(* y: 'c (fresh) *)",
                    "(* Constraint: 'b = 'c → 'd *)",
                ],
            ),
            (
                "f_y",
                [
                    "(* Environment: *)",
                    "(* x: 'a (fresh) *)",
                    "(* f: 'c → 'd *)",
                    "(* y: 'c *)",
                    "(* Result type: 'd *)",
                ],
            ),
            (
                "x",
                [
                    "(* Unifying constraints of x: *)",
                    "(* x takes parameters: f: 'c → 'd, y: 'c *)",
                    "(* x returns: 'd *)",
                    "(* x's type: ('c → 'd) → 'c → 'd *)",
                ],
            ),
        ]

        env_block: VGroup | None = None

        for part, lines in steps:
            if part is not None:
                self.play(code.indicate(part))

            new_block = build_env_block(lines)

            if env_block is None:
                self.play(FadeIn(new_block, shift=UP))
            else:
                # Build animations per corresponding line
                anims = []
                common = min(len(env_block), len(new_block))
                for i in range(common):
                    anims.append(ReplacementTransform(env_block[i], new_block[i]))
                # Handle removed lines
                for i in range(common, len(env_block)):
                    anims.append(FadeOut(env_block[i]))
                # Handle added lines
                for i in range(common, len(new_block)):
                    anims.append(FadeIn(new_block[i], shift=UP))
                self.play(AnimationGroup(*anims, lag_ratio=0))

            self.wait(1)
            env_block = new_block

        # Pause a bit longer before showing the final inferred type
        self.wait(2)

        # Final block
        final_lines = [
            "(* val x : ('c → 'd) → 'c → 'd *)",
        ]
        final_block = build_env_block(final_lines)

        anims = []
        common = min(len(env_block), len(final_block))
        for i in range(common):
            anims.append(ReplacementTransform(env_block[i], final_block[i]))
        for i in range(common, len(env_block)):
            anims.append(FadeOut(env_block[i]))
        for i in range(common, len(final_block)):
            anims.append(FadeIn(final_block[i], shift=UP))
        self.play(AnimationGroup(*anims, lag_ratio=0))
        self.wait()


if __name__ == "__main__":
    # This section is used when the file is run directly
    pass
