#!/usr/bin/env python3
"""Draw the bedtools-style cartoon of how flank_metaplot.py bins a bedGraph.

Two panels:
  1. --value: input ranges (any width, gaps, sub-window) -> per-window
     overlap-weighted mean, uncovered bp = 0.
  2. --event: overlapping event ranges -> per-bp densities (value/len) ADD;
     per-window value integrates that density.
Outputs docs/how_it_works.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "how_it_works.png")

IN_A = "#4c9bd5"   # input track colour
IN_B = "#f2a900"   # second/overlapping input colour
RES = "#8bc34a"    # result (green), bedtools-style
GRID = "0.55"


def box(ax, x0, x1, y, h, color, label=None, fs=11):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=color,
                           edgecolor="black", lw=2, joinstyle="round"))
    if label:
        ax.text((x0 + x1) / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def windows(ax, edges, ytop, label_y):
    for e in edges:
        ax.axvline(e, ymin=0.05, ymax=0.95, ls="--", lw=1.5, color=GRID)
    for i in range(len(edges) - 1):
        ax.text((edges[i] + edges[i + 1]) / 2, label_y, f"window {i+1}",
                ha="center", va="center", fontsize=10, color=GRID)


def main():
    plt.xkcd()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.6))

    # ---------- panel 1: --value (box HEIGHT is proportional to the value) ----
    edges = [0, 5, 10, 15]
    windows(ax1, edges, 3.0, 3.5)
    SCALE = 0.22                       # plot height per unit value
    in_base, res_base = 1.55, 0.15
    ax1.text(-0.3, in_base + 0.55, "input", ha="right", va="center", fontsize=11)
    ax1.text(-0.3, res_base + 0.35, "per window", ha="right", va="center", fontsize=11)
    for b in (in_base, res_base):     # faint baselines (value = height above these)
        ax1.plot([0, 15], [b, b], color="0.75", lw=1, zorder=0)
    # input ranges: value held at every bp; height = value (sub-window range + a gap)
    for x0, x1, v in [(0, 3, 2), (3.4, 4.4, 6), (6, 12, 3)]:
        box(ax1, x0, x1, in_base, v * SCALE, IN_A)
        ax1.text((x0 + x1) / 2, in_base + v * SCALE + 0.08, f"v={v}",
                 ha="center", va="bottom", fontsize=10)
    ax1.text(13.5, in_base + 0.12, "gap→0", ha="center", va="bottom", fontsize=8, color="0.4")
    # per-window overlap-weighted mean; height = value; uncovered bp count as 0
    for x0, x1, v in [(0, 5, 2.4), (5, 10, 2.4), (10, 15, 1.2)]:   # (2·3+6·1)/5, 3·4/5, 3·2/5
        box(ax1, x0, x1, res_base, v * SCALE, RES)
        ax1.text((x0 + x1) / 2, res_base + v * SCALE / 2, f"{v}",
                 ha="center", va="center", fontsize=10)
    ax1.set_title("--value:  window = Σ(valueᵢ × overlapᵢ) / window width   "
                  "(box height = value; uncovered bp = 0)", fontsize=12)
    ax1.set_xlim(-2.4, 15.5); ax1.set_ylim(0, 4.2)

    # ---------- panel 2: --event (bar HEIGHT is the per-bp density value/len) --
    windows(ax2, [0, 5, 10], 4.0, 4.6)
    ESCALE = 2.6
    ax2.text(-0.3, 3.8, "events", ha="right", va="center", fontsize=11)
    ax2.text(-0.3, 1.75, "density\n(Σ value/len)", ha="right", va="center", fontsize=9)
    ax2.text(-0.3, 0.45, "per window", ha="right", va="center", fontsize=11)
    # two overlapping crossovers, each = 1 event; height = its per-bp density
    for x0, x1, d, lab, base in [(1, 7, 1 / 6, "1 event (1/6 per bp)", 3.6),
                                 (4, 9, 1 / 5, "1 event (1/5 per bp)", 2.7)]:
        box(ax2, x0, x1, base, d * ESCALE, IN_B)
        ax2.text((x0 + x1) / 2, base + d * ESCALE + 0.06, lab, ha="center",
                 va="bottom", fontsize=8)
    # flattened per-bp density: overlapping ranges ADD (height = summed density)
    for a, b, d in [(1, 4, 1 / 6), (4, 7, 1 / 6 + 1 / 5), (7, 9, 1 / 5)]:
        box(ax2, a, b, 1.4, d * ESCALE, RES)
    ax2.text(5.5, 1.4 + (1 / 6 + 1 / 5) * ESCALE + 0.06, "overlap: densities add",
             ha="center", va="bottom", fontsize=8, color="0.35")
    # per-window value = integral of density / window width; height = that value
    for x0, x1, val in [(0, 5, 0.173), (5, 10, 0.227)]:   # ∫density/width
        box(ax2, x0, x1, 0.15, val * ESCALE, RES)
        ax2.text((x0 + x1) / 2, 0.15 + val * ESCALE / 2, f"{val:.2f}",
                 ha="center", va="center", fontsize=9)
    ax2.set_title("--event:  count spread over each range (value/len per bp); "
                  "bar height = density; overlaps add", fontsize=12)
    ax2.set_xlim(-2.4, 15.5); ax2.set_ylim(0, 5.0)

    for ax in (ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
