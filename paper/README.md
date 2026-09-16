# Paper

Reverse Engineering the RVQ Audio Encoder of MiniMax Music 3 by Reverse Distillation.

Every figure and table is generated from the committed results, so the build is two commands
from a clean checkout:

    uv run python paper/tables.py       # results/ and benchmarks/ into paper/tables/ and docs/
    uv run python paper/figures.py      # results/ and benchmarks/ into paper/figures/*.pdf
    tectonic paper/main.tex             # paper/main.pdf

tectonic 0.15.0 was used for the committed PDF; any TeX Live with pdflatex, natbib and tikz
builds the same source. The pipeline and timeline diagrams are TikZ in paper/figures/*.tex.
