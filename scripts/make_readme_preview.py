"""Draw the illustrated README walkthrough. Requires Pillow."""

import argparse
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, SCALE = 1200, 760, 2
PAPER, WHITE, INK = "#f5f3ed", "#fffefb", "#252521"
MUTED, LINE, LIME = "#6c6a66", "#dad8d0", "#d7fb61"
VIOLET, LAVENDER = "#6940d3", "#e8e1f6"
QUESTION = "How can RAG reduce hallucinations in medical question answering?"
PAPERS = (
    "Rationale-Guided Retrieval Augmented Generation for Medical Question Answering",
    "Self-MedRAG: a Self-Reflective Hybrid Retrieval-Augmented Generation Framework for Reliable Medical Question Answering",
)


@lru_cache(maxsize=None)
def font(size, weight="regular"):
    candidates = {
        "regular": ["/System/Library/Fonts/Avenir.ttc", "DejaVuSans.ttf"],
        "bold": ["/System/Library/Fonts/Avenir Next.ttc", "DejaVuSans-Bold.ttf"],
        "mono": ["/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"],
    }
    for candidate in candidates[weight]:
        try:
            return ImageFont.truetype(candidate, size * SCALE)
        except OSError:
            continue
    raise RuntimeError("Install Avenir or DejaVu fonts to draw the preview.")


class Canvas:
    def __init__(self):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), PAPER)
        self.draw = ImageDraw.Draw(self.image)

    def box(self, x, y, width, height, fill, radius=12, outline=None, stroke=1):
        self.draw.rounded_rectangle(
            tuple(round(value * SCALE) for value in (x, y, x + width, y + height)),
            radius=round(radius * SCALE), fill=fill, outline=outline,
            width=max(1, round(stroke * SCALE)),
        )

    def line(self, points, fill=INK, width=1):
        self.draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
                       fill=fill, width=max(1, round(width * SCALE)), joint="curve")

    def text(self, x, y, value, size=18, color=INK, weight="regular"):
        self.draw.text((round(x * SCALE), round(y * SCALE)), value,
                       font=font(size, weight), fill=color, anchor="lt")

    def paragraph(self, x, y, value, width, size=18, color=MUTED, weight="regular", leading=None):
        lines, current = [], ""
        for word in value.split():
            candidate = f"{current} {word}".strip()
            if font(size, weight).getlength(candidate) > width * SCALE and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for index, row in enumerate(lines):
            self.text(x, y + index * (leading or size * 1.35), row, size, color, weight)
        return len(lines) * (leading or size * 1.35)

    def arrow(self, x, y, size=13, color=INK):
        self.line([(x, y + size), (x + size, y), (x + size, y + size * .7)], color, 2)
        self.line([(x + size * .3, y), (x + size, y)], color, 2)

    def button(self, x, y, width, label, active=False):
        self.box(x, y + 3, width, 43, INK, 9)
        self.box(x, y, width, 43, "#e8ff9f" if active else LIME, 9, INK)
        self.text(x + 14, y + 13, label, 15, INK, "bold")
        self.arrow(x + width - 29, y + 15, 12)

    def star(self, x, y, radius, angle=0):
        for index in range(8):
            theta = angle + index * math.pi / 4
            self.line([(x + math.cos(theta) * radius * .23, y + math.sin(theta) * radius * .23),
                       (x + math.cos(theta) * radius, y + math.sin(theta) * radius)], VIOLET, 8)

    def finish(self):
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def paper_card(canvas, y, number, title, suggested=False):
    canvas.box(438, y, 660, 91, WHITE, 12, LINE)
    canvas.box(453, y + 15, 27, 27, LAVENDER, 6)
    canvas.text(460, y + 23, str(number).zfill(2), 11, VIOLET, "mono")
    canvas.text(493, y + 17, "arXiv", 11, VIOLET, "mono")
    if suggested:
        canvas.box(1002, y + 12, 79, 24, LIME, 5)
        canvas.text(1014, y + 19, "Shortlist", 10, INK, "bold")
    canvas.paragraph(493, y + 38, title, 580, 16, INK, "bold", leading=21)


def frame(phase, progress):
    c = Canvas()
    c.box(18, 18, 1164, 724, PAPER, 25, "#c9c5ba")
    c.box(45, 45, 44, 44, LIME, 12, INK)
    c.line([(58, 56), (73, 56), (79, 62), (79, 79), (58, 79), (58, 56)], INK, 2)
    c.line([(73, 56), (73, 63), (79, 63)], INK, 2)
    c.line([(63, 68), (73, 68)], INK, 2)
    c.line([(63, 73), (70, 73)], INK, 2)
    c.text(102, 48, "paper retriever", 27, INK, "bold")
    c.text(103, 80, "FOLLOW YOUR CURIOSITY", 10, MUTED, "mono")
    c.box(852, 52, 300, 32, LAVENDER, 16)
    c.text(871, 64, "AN ILLUSTRATED WALKTHROUGH", 11, VIOLET, "mono")
    c.line([(45, 109), (1152, 109)], LINE)
    c.text(50, 137, "Go down a better rabbit hole.", 45, INK, "bold")
    c.line([(314, 189), (693, 185), (724, 189)], LIME, 8)
    c.text(52, 205, "A clearer question. A search you guide. A reading list to explore.", 18, MUTED)
    c.star(1100, 167, 31, progress * .15 + phase * .2)

    c.box(49, 255, 318, 408, WHITE, 18, "#bcb8ad")
    c.text(71, 279, "YOUR STARTING POINT", 11, MUTED, "mono")
    c.text(70, 307, "Start with a spark.", 26, INK, "bold")
    c.box(69, 351, 278, 119, "#faf9f5", 10, LINE)
    c.paragraph(83, 367, QUESTION, 249, 18, INK, leading=25)
    c.text(72, 493, "YOUR LOCAL MODEL", 10, MUTED, "mono")
    c.box(70, 516, 276, 37, "#faf9f5", 8, LINE)
    c.text(82, 527, "qwen3:4b", 14, INK, "mono")
    c.text(72, 571, "3 search rounds · 5 papers each", 12, MUTED)
    c.button(70, 605, 276, "Start new research" if phase == 4 else "Research in progress")

    c.box(391, 255, 760, 408, "#eae4f3", 18, "#d2c8e2")
    current_stage = 0 if phase == 0 else 1 if phase < 4 else 2
    for index, title in enumerate(("Refine question", "Explore papers", "Review shortlist")):
        x = 411 + index * 240
        c.box(x, 273, 230, 39, WHITE if index == current_stage else "#eee9f5", 9)
        c.box(x + 9, 282, 21, 21, LIME if index == current_stage else "#ded5ed", 10)
        c.text(x + 14, 289, str(index + 1), 10, INK, "mono")
        c.text(x + 39, 286, title, 13, INK if index == current_stage else MUTED, "bold")

    c.box(412, 329, 718, 312, WHITE, 14, "#d1c7e0")
    labels = ["01 / SHARPEN THE QUESTION", "02 / APPROVE THE SEARCH", "03 / REVIEW THE EVIDENCE",
              "04 / REFINE THE NEXT SEARCH", "05 / YOUR READING LIST"]
    headings = ["Make the question yours.", "Your search. Your call.", "Follow what turns up.",
                "The papers shape the next step.", "Your reading list is ready."]
    c.text(437, 350, labels[phase], 11, VIOLET, "mono")
    c.text(435, 379, headings[phase], 27, INK, "bold")

    if phase == 0:
        c.box(436, 428, 668, 92, "#faf9f5", 9, LINE)
        c.paragraph(451, 445, "Which retrieval methods reduce hallucinations in medical question answering, and how are they evaluated?",
                    634, 19, INK, leading=27)
        for x, width, label, color in ((436, 110, "Medical QA", LAVENDER), (555, 101, "Retrieval", "#e9efcf"), (665, 113, "Evaluation", "#fae3d5")):
            c.box(x, 535, width, 29, color, 6)
            c.text(x + 12, 544, label, 12, INK)
        c.text(438, 596, "Review or edit the wording.", 13, MUTED)
        c.button(867, 579, 237, "Approve question", progress > .65)
    elif phase in (1, 3):
        c.text(438, 432, "Proposed search keywords", 13, MUTED)
        c.box(436, 460, 668, 55, "#faf9f5", 9, VIOLET)
        query = "RAG hallucination medical QA evaluation" if phase == 1 else "medical RAG hallucination benchmarks"
        c.text(452, 481, query, 17, INK, "mono")
        reason = "You can change these keywords before they go to arXiv." if phase == 1 else "The agent can look for more evidence within your search limit."
        c.paragraph(438, 533, reason, 630, 15, MUTED)
        c.text(438, 599, "You approve every search.", 13, MUTED)
        c.button(867, 579, 237, "Approve & search", progress > .65)
    else:
        paper_card(c, 426, 1, PAPERS[0], suggested=phase == 4)
        paper_card(c, 526, 2, PAPERS[1], suggested=phase == 4)

    c.box(414, 678, 715, 4, "#e2dbeb", 2)
    c.box(414, 678, 715 * (phase + .12 + progress * .88) / 5, 4, VIOLET, 2)
    captions = ["Review the question before searching.", "Search happens after your approval.",
                "The model reviews titles and abstract excerpts.", "New evidence can lead to another query.",
                "Keep the papers, questions, and search history."]
    c.text(51, 703, captions[phase], 14, MUTED)
    c.box(952, 697, 195, 24, LIME, 12)
    c.text(968, 705, "HUMAN IN THE LOOP", 11, INK, "mono")
    return c.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs/images")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [frame(phase, step / 9) for phase in range(5) for step in range(10)]
    poster = frame(4, .5)
    poster.save(args.output_dir / "walkthrough.png", optimize=True)

    colors = Image.new("RGB", (600, 380 * 5))
    for phase in range(5):
        colors.paste(frames[phase * 10].resize((600, 380)), (0, phase * 380))
    palette = colors.quantize(colors=256)
    indexed = [item.quantize(palette=palette, dither=Image.Dither.NONE) for item in frames]
    durations = [1500 if index % 10 == 9 else 170 for index in range(len(indexed))]
    indexed[0].save(args.output_dir / "walkthrough.gif", save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, disposal=1, optimize=True)
    print(f"Saved walkthrough.gif and walkthrough.png to {args.output_dir}")


if __name__ == "__main__":
    main()
