#!/usr/bin/env python3
"""Generic per-gene metaplot for one or two bedGraph value/event columns.

Input is one or two bedGraph/BED files (--bed, repeatable up to 2). Columns to
plot are selected with:
  --value COL[,COL]  a "value" track: the column value is held at every bp of
                     the range (a rate / level, e.g. cM/Mb or pi).
  --event COL[,COL]  an "event" track: the column counts events spread uniformly
                     over the range, so the per-bp density is value / range_bp
                     (e.g. 3 crossovers in a 1500 bp range -> 3/1500 per bp).
Columns bind to the most recent --bed, so with two files each contributes its
own column(s). At most two columns total are plotted; with two, the first is
drawn on the left y-axis and the second on a right (twin) axis.

Each range is piecewise-constant over the bp it spans. A window's value is the
overlap-weighted integral of the per-bp density divided by the full window
width X:
    value = sum_i(density_i * overlap_i) / X
with uncovered bp counted as zero (a window with no coverage at all is skipped,
not zeroed). density_i is the column value (--value) or value/range_bp (--event).

Values are summarised around gene edges (mean +-1 SE box per slot) in:
  * linear 5' / 3' flanks (windows of --win out to --flank-bp),
  * the gene body (bins of --win from TSS and TTS, split at the gene midpoint),
  * optional far-field boxes at --box-dists (e.g. 10 / 50 / 100 kb).

Layout (left -> right, mirrored):
  [far boxes] | 5' flank | TSS |body| gap |body| TTS | 3' flank | [far boxes]
The bottom panel shows the number of genes contributing to each slot (track 1).
Leading bedGraph track / browser / # header lines are ignored.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metaplot as mp  # noqa: E402

DEFAULT_COLORS = ["#1f77b4", "#d62728"]


class SeriesAction(argparse.Action):
    """Collect --value / --event columns in command-line order, each bound to
    the most recent --bed (so `--bed A --event 4 --bed B --value 4` works)."""

    def __call__(self, parser, namespace, values, option_string=None):
        spec = getattr(namespace, "series", None) or []
        beds = getattr(namespace, "bed", None) or []
        kind = "event" if option_string == "--event" else "value"
        cols = []
        for v in values:
            cols += str(v).replace(",", " ").split()
        for c in cols:
            spec.append({"kind": kind, "col": int(c), "bed": len(beds) - 1})
        namespace.series = spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gff", required=True,
                   help="GFF3 of gene annotations; 'gene' features define the TSS/TTS "
                        "edges and strand that the profile is built around.")
    p.add_argument("--bed", action="append", required=True, metavar="BED",
                   help="Input bedGraph/BED. Repeat once for a second file (max 2). "
                        "--value/--event columns bind to the most recent --bed.")
    p.add_argument("--value", action=SeriesAction, nargs="+", metavar="COL",
                   help="1-based column(s) plotted as a VALUE track (value held at "
                        "every bp of the range). Space- or comma-separated.")
    p.add_argument("--event", action=SeriesAction, nargs="+", metavar="COL",
                   help="1-based column(s) plotted as an EVENT track (per-bp density "
                        "= value / range_bp). Space- or comma-separated.")
    p.add_argument("--label", action="append", default=None, metavar="TEXT",
                   help="Legend label per series (in command-line order).")
    p.add_argument("--ylabel", action="append", default=None, metavar="TEXT",
                   help="Y-axis label per series (in command-line order).")
    p.add_argument("--plot_color", nargs="+", default=None, metavar="COLOR",
                   help="Box colour(s) per series, in command-line order. Default blue, red.")
    p.add_argument("--gene_color", default="black", metavar="COLOR",
                   help="Colour of the bottom gene-count bars (default black).")
    p.add_argument("--legend-loc", default="upper right",
                   choices=["best", "upper right", "upper left", "lower left",
                            "lower right", "right", "center left", "center right",
                            "lower center", "upper center", "center", "none"],
                   help="Legend position (matplotlib loc), or 'none' to hide it "
                        "(default 'upper right').")
    p.add_argument("--flank-bp", type=int, default=5000,
                   help="Number of bp upstream and downstream of each gene to profile "
                        "in the linear flanks (default 5000).")
    p.add_argument("--win", type=int, default=500,
                   help="Window/bin size in bp for the flank windows and gene-body bins "
                        "(default 500).")
    p.add_argument("--body-bins", type=int, default=3,
                   help="Number of --win bins profiled inward from each of the TSS and "
                        "TTS into the gene body (default 3); the gene interior beyond "
                        "these bins (past the gene midpoint) is not sampled.")
    p.add_argument("--box-dists", type=int, nargs="*", default=[10000, 50000, 100000],
                   help="Centre distances (bp) from the gene edge for the far-field "
                        "summary boxes, drawn on each side (default 10000 50000 100000). "
                        "Any number of distances may be given (sorted automatically); "
                        "pass no values to disable the far-field boxes.")
    p.add_argument("--box-halfwidth", type=int, default=250,
                   help="Half-width (bp) of each far-field box: the box spans its centre "
                        "distance +- this many bp (default 250 = a 500bp window at that "
                        "distance).")
    p.add_argument("--output", default="flank_metaplot.pdf",
                   help="Output figure path; the extension sets the format, e.g. .pdf / "
                        ".png / .svg (default flank_metaplot.pdf).")
    p.add_argument("--title", default="Metaplot around genes",
                   help="Title printed above the plot (default 'Metaplot around genes').")
    return p.parse_args()


# ---------------------------------------------------------------- loading ----
def is_header(line):
    s = line.lstrip()
    low = s.lower()
    return (not s.strip()) or s.startswith("#") or low.startswith(("track", "browser"))


def header_rows(path):
    """Indices of leading bedGraph header lines (track / browser / # / blank)."""
    skip = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if is_header(line):
                skip.append(i)
            else:
                break
    return skip


def load_bed_df(path):
    df = pd.read_csv(path, sep="\t", header=None, comment="#", skiprows=header_rows(path))
    df = df.rename(columns={0: "chr", 1: "start", 2: "end"})
    df["chr"] = df["chr"].map(mp.normalize_chrom_name)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    return df


def build_track(df, col, kind):
    """Build a per-chromosome step-function track from one column.

    Stores, per chrom, ranges sorted by start plus the cumulative integral of two
    channels so any interval integral is two O(log n) lookups (see integrate()):
      cov  -> coverage (per-bp density 1); used to detect empty windows.
      val  -> the plotted quantity's integral over each range:
                value * length   for a VALUE track (density = value), or
                value            for an EVENT track (density = value / length).
    """
    c = col - 1
    if c not in df.columns:
        raise SystemExit(f"column {col} not present in input ({len(df.columns)} columns)")
    sub = df[["chr", "start", "end", c]].copy()
    sub.columns = ["chr", "start", "end", "v"]
    sub["v"] = pd.to_numeric(sub["v"], errors="coerce")
    sub = sub.dropna(subset=["chr", "start", "end", "v"])
    sub["start"] = sub["start"].astype(np.int64); sub["end"] = sub["end"].astype(np.int64)
    sub = sub[sub["end"] > sub["start"]]
    out = {}
    for chrom, gsub in sub.groupby("chr", sort=False):
        gsub = gsub.sort_values("start", kind="mergesort")
        s = gsub["start"].to_numpy(np.int64); e = gsub["end"].to_numpy(np.int64)
        # The overlap integration assumes a proper (non-overlapping) bedGraph;
        # overlapping ranges would silently give wrong per-window values.
        bad = np.nonzero(s[1:] < e[:-1])[0]
        if bad.size:
            i = bad[0]
            raise SystemExit(
                f"overlapping ranges in column {col} on {chrom}: "
                f"[{s[i]},{e[i]}) overlaps [{s[i+1]},{e[i+1]}). Input must be a "
                f"non-overlapping bedGraph; flatten overlapping intervals first.")
        length = (e - s).astype(np.float64)
        v = gsub["v"].to_numpy(np.float64)
        val_integ = v * length if kind == "value" else v.copy()
        integ = {"cov": length.copy(), "val": val_integ}
        cum = {k: np.concatenate(([0.0], np.cumsum(arr))) for k, arr in integ.items()}
        out[chrom] = dict(s=s, e=e, length=length, integ=integ, cum=cum)
    return dict(chroms=out, kind=kind, names=["cov", "val"])


def integrate(chrom, q, names):
    """G(x) = integral of each channel's per-bp density over (-inf, x], at points q.

    An interval integral is G(hi) - G(lo). Gaps between ranges contribute 0.
    """
    s = chrom["s"]; e = chrom["e"]; length = chrom["length"]
    q = np.asarray(q, dtype=np.float64)
    j = np.searchsorted(s, q, side="right") - 1      # last range starting at/left of q
    valid = j >= 0
    jj = np.clip(j, 0, len(s) - 1)
    inside = valid & (q < e[jj])                      # q within range jj (else in a gap after it)
    out = {}
    for name in names:
        integ = chrom["integ"][name]; cum = chrom["cum"][name]
        base = cum[jj]                                # full integral of ranges before jj
        dens = integ[jj] / length[jj]
        G = np.where(inside, base + dens * (q - s[jj]), base + integ[jj])
        G[~valid] = 0.0
        out[name] = G
    return out


# ------------------------------------------------------------ accumulators ----
def zarr(n):
    return {"s": np.zeros(n), "q": np.zeros(n), "n": np.zeros(n, np.int64)}


def region_vals(chrom, lo, hi, track):
    """Per-bin value for a set of [lo, hi) windows: integral(val)/width, gaps=0.

    Returns (values, ok) where ok masks windows with positive width and coverage.
    """
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    ext = hi - lo
    n = len(lo)
    g = integrate(chrom, np.concatenate([lo, hi]), track["names"])
    cov = g["cov"][n:] - g["cov"][:n]
    valint = g["val"][n:] - g["val"][:n]
    val = np.divide(valint, ext, out=np.full_like(ext, np.nan), where=ext > 0)
    ok = (ext > 0) & (cov > 0)
    return val, ok


def acc_add(acc, idx, val, ok):
    idx = np.asarray(idx)[ok]; v = val[ok]
    np.add.at(acc["s"], idx, v)
    np.add.at(acc["q"], idx, v * v)
    np.add.at(acc["n"], idx, 1)


def stats(acc):
    s, q, n = acc["s"], acc["q"], acc["n"]
    m = np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
    v = np.maximum(np.divide(q, n, out=np.zeros_like(s), where=n > 0) - np.nan_to_num(m) ** 2, 0)
    se = np.full_like(s, np.nan); se[n > 0] = np.sqrt(v[n > 0] / n[n > 0])
    return m, se, n


# --------------------------------------------------------------- geometry ----
def build_layout(nflank, nbody, nbox, gap=1.0, interior=1.2):
    """Return x-positions for every slot, adapting to bin counts."""
    c = 0.0
    box_left = np.zeros(nbox)      # index 0 = nearest box distance
    for j in range(nbox):          # place farthest first (left), nearest last
        box_left[nbox - 1 - j] = c; c += 1
    if nbox:
        sep_left = c - 0.5 + gap / 2; c += gap
    else:
        sep_left = None
    flank_up = np.zeros(nflank)    # bin 0 = nearest edge (rightmost)
    for b in range(nflank - 1, -1, -1):
        flank_up[b] = c; c += 1
    x_tss = c - 0.5
    body_tss = np.arange(c, c + nbody); c += nbody
    interior_lo = c - 0.5; c += interior
    interior_hi = c - 0.5
    body_tts = np.zeros(nbody)     # bin 0 = nearest TTS (rightmost)
    for b in range(nbody - 1, -1, -1):
        body_tts[b] = c; c += 1
    x_tts = c - 0.5
    flank_dn = np.arange(c, c + nflank); c += nflank
    if nbox:
        sep_right = c - 0.5 + gap / 2; c += gap
    else:
        sep_right = None
    box_right = np.arange(c, c + nbox); c += nbox  # index 0 = nearest distance
    return dict(box_left=box_left, flank_up=flank_up, x_tss=x_tss,
                body_tss=body_tss, interior=(interior_lo, interior_hi),
                body_tts=body_tts, x_tts=x_tts, flank_dn=flank_dn,
                sep_left=sep_left, sep_right=sep_right, box_right=box_right,
                xmax=c - 1)


# ------------------------------------------------------------------- main ----
def main():
    args = parse_args()
    beds = args.bed
    series = getattr(args, "series", None) or []
    if not series:
        raise SystemExit("Provide at least one --value or --event column.")
    if len(series) > 2:
        raise SystemExit("At most two columns total (--value/--event) can be plotted.")
    if len(beds) > 2:
        raise SystemExit("At most two --bed files.")
    for sp in series:
        if sp["bed"] < 0:
            if len(beds) == 1:
                sp["bed"] = 0
            else:
                raise SystemExit("Each --value/--event must come after a --bed.")

    win = args.win
    nflank = args.flank_bp // win
    flank_extent = nflank * win          # actual bp profiled (flank_bp rounded down to a whole win)
    if args.flank_bp % win:
        print(f"Warning: --flank-bp {args.flank_bp} is not a multiple of --win {win}; "
              f"flanks profiled to {flank_extent} bp (last {args.flank_bp % win} bp dropped).",
              file=sys.stderr)
    nbody = args.body_bins
    box_d = np.array(sorted(args.box_dists), dtype=np.float64)
    nbox = len(box_d)
    hw = args.box_halfwidth

    bed_dfs = {}
    for sp in series:
        if sp["bed"] not in bed_dfs:
            bed_dfs[sp["bed"]] = load_bed_df(beds[sp["bed"]])

    def opt(lst, i):
        return lst[i] if lst and i < len(lst) else None

    tracks = []
    for i, sp in enumerate(series):
        tr = build_track(bed_dfs[sp["bed"]], sp["col"], sp["kind"])
        stem = os.path.basename(beds[sp["bed"]]).split(".")[0]
        lab = opt(args.label, i) or f"{stem} col{sp['col']} ({sp['kind']})"
        tr.update(
            label=lab,
            ylabel=opt(args.ylabel, i) or lab,
            color=opt(args.plot_color, i) or DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            fl={s: zarr(nflank) for s in ("up", "dn")},
            bd={e: zarr(nbody) for e in ("tss", "tts")},
            bx={s: zarr(nbox) for s in ("up", "dn")},
        )
        tracks.append(tr)

    genes = mp.load_genes(args.gff).sort_values(["chr", "start"], kind="mergesort").reset_index(drop=True)
    genes["prev_end"] = np.int64(-1)
    genes["next_start"] = np.iinfo(np.int64).max
    for _, idx in genes.groupby("chr", sort=False).groups.items():
        sub = genes.loc[idx].sort_values("start", kind="mergesort")
        ends = sub["end"].to_numpy(np.int64); starts = sub["start"].to_numpy(np.int64)
        pe = np.full(len(sub), -1, np.int64); ns = np.full(len(sub), np.iinfo(np.int64).max, np.int64)
        if len(sub) > 1:
            pe[1:] = np.maximum.accumulate(ends[:-1]); ns[:-1] = starts[1:]
        genes.loc[sub.index, "prev_end"] = pe
        genes.loc[sub.index, "next_start"] = ns

    bfl = np.arange(nflank); bbx = np.arange(nbox); bbd = np.arange(nbody)

    for _, g in genes.iterrows():
        chrom = g["chr"]
        gs, ge = int(g["start"]), int(g["end"])
        pe, ns = int(g["prev_end"]), int(g["next_start"])
        # clip flanks/boxes at the midpoint to the neighbouring gene (and >=0)
        lo_left = max((pe + gs) / 2.0, 0.0) if pe >= 0 else 0.0
        hi_right = (ge + ns) / 2.0 if ns < np.iinfo(np.int64).max else np.inf
        gene_mid = (gs + ge) / 2.0

        if g["strand"] == "+":
            sides = [("up", gs, "left"), ("dn", ge, "right")]
            body_edges = [("tss", gs, "right"), ("tts", ge, "left")]
        else:
            sides = [("up", ge, "right"), ("dn", gs, "left")]
            body_edges = [("tss", ge, "left"), ("tts", gs, "right")]

        for t in tracks:
            cd = t["chroms"].get(chrom)
            if cd is None:
                continue
            # ---- flanks + far boxes (outward from each edge) ----
            for tag, edge, outward in sides:
                if outward == "left":
                    fl_hi = edge - bfl * win
                    fl_lo = np.maximum(edge - (bfl + 1) * win, lo_left)
                    bx_hi = edge - (box_d - hw)
                    bx_lo = np.maximum(edge - (box_d + hw), lo_left)
                else:
                    fl_lo = edge + bfl * win
                    fl_hi = np.minimum(edge + (bfl + 1) * win, hi_right)
                    bx_lo = edge + (box_d - hw)
                    bx_hi = np.minimum(edge + (box_d + hw), hi_right)
                if nflank:
                    val, ok = region_vals(cd, fl_lo, fl_hi, t)
                    acc_add(t["fl"][tag], bfl, val, ok)
                if nbox:
                    val, ok = region_vals(cd, bx_lo, bx_hi, t)
                    acc_add(t["bx"][tag], bbx, val, ok)
            # ---- gene body: bins from each edge, split at the gene midpoint ----
            for edge_name, e_bp, direction in body_edges:
                if direction == "right":                       # inward = increasing coord
                    lo = np.maximum(e_bp + bbd * win, gs)
                    hi = np.minimum(e_bp + (bbd + 1) * win, gene_mid)
                else:                                          # inward = decreasing coord
                    hi = np.minimum(e_bp - bbd * win, ge)
                    lo = np.maximum(e_bp - (bbd + 1) * win, gene_mid)
                val, ok = region_vals(cd, lo, hi, t)
                acc_add(t["bd"][edge_name], bbd, val, ok)

    # ---------------------------------------------------------- plotting ----
    L = build_layout(nflank, nbody, nbox)
    fig, (ax, ax_c) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    axes = [ax, ax.twinx()] if len(tracks) == 2 else [ax]
    BW = 0.34

    def boxes(axis, xs, acc, color, w=BW):
        m, se, _ = stats(acc)
        for xi, mi, si in zip(np.atleast_1d(xs), m, se):
            if not np.isfinite(mi):
                continue
            si = si if np.isfinite(si) else 0.0
            axis.add_patch(Rectangle((xi - w / 2, mi - si), w, 2 * si,
                                     facecolor=color, alpha=0.35, edgecolor=color, lw=0.7, zorder=4))
            axis.plot([xi - w / 2, xi + w / 2], [mi, mi], color=color, lw=1.2, zorder=5)

    for t, axis in zip(tracks, axes):
        c = t["color"]
        boxes(axis, L["flank_up"], t["fl"]["up"], c)
        boxes(axis, L["flank_dn"], t["fl"]["dn"], c)
        boxes(axis, L["body_tss"], t["bd"]["tss"], c)
        boxes(axis, L["body_tts"], t["bd"]["tts"], c)
        if nbox:
            boxes(axis, L["box_left"], t["bx"]["up"], c)
            boxes(axis, L["box_right"], t["bx"]["dn"], c)

    # gene shading + separators
    for a in (ax, ax_c):
        a.axvspan(L["x_tss"], L["x_tts"], color="#ececec", alpha=0.8, zorder=0)
        a.axvspan(L["interior"][0], L["interior"][1], color="#c8c8c8", alpha=0.9, zorder=0)
        a.axvline(L["x_tss"], ls="--", lw=1, color="0.3")
        a.axvline(L["x_tts"], ls="--", lw=1, color="0.3")
        for sp in (L["sep_left"], L["sep_right"]):
            if sp is not None:
                a.axvline(sp, ls=":", lw=1, color="0.6")

    axes[0].set_ylabel(tracks[0]["ylabel"], color=tracks[0]["color"])
    axes[0].tick_params(axis="y", labelcolor=tracks[0]["color"])
    if len(tracks) == 2:
        axes[1].set_ylabel(tracks[1]["ylabel"], color=tracks[1]["color"])
        axes[1].tick_params(axis="y", labelcolor=tracks[1]["color"])
    ax.set_title(args.title)

    leg = [Patch(facecolor=t["color"], alpha=0.5, edgecolor=t["color"], label=t["label"])
           for t in tracks]
    leg += [Patch(facecolor="#ececec", edgecolor="0.6", label="gene body (TSS–TTS)"),
            Patch(facecolor="#c8c8c8", edgecolor="0.6", label="gene interior (not sampled)")]
    if args.legend_loc != "none":
        legend = ax.legend(handles=leg, loc=args.legend_loc, fontsize=7, framealpha=0.92,
                           title="boxes = mean ± 1 SE", title_fontsize=7)
        legend.get_frame().set_edgecolor("none")

    # ---------- bottom bar panel: genes contributing (first track) ----------
    t0 = tracks[0]

    def bar(xs, acc):
        _, _, n = stats(acc)
        ax_c.bar(xs, n, width=0.85, color=args.gene_color)
    if nbox:
        bar(L["box_left"], t0["bx"]["up"])
        bar(L["box_right"], t0["bx"]["dn"])
    bar(L["flank_up"], t0["fl"]["up"])
    bar(L["flank_dn"], t0["fl"]["dn"])
    bar(L["body_tss"], t0["bd"]["tss"])
    bar(L["body_tts"], t0["bd"]["tts"])
    ax_c.set_ylabel("Genes\ncontributing", fontsize=8)
    ax_c.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax_c.tick_params(axis="y", labelsize=7)

    # ticks
    def kb(v):
        return f"{v / 1000:g}kb"
    xticks, xlabels = [], []
    for dist in sorted(box_d, reverse=True):
        xticks.append(L["box_left"][np.where(box_d == dist)[0][0]]); xlabels.append(kb(dist))
    xticks += [L["flank_up"][-1], L["x_tss"], L["x_tts"], L["flank_dn"][-1]]
    xlabels += [f"-{kb(flank_extent)}", "TSS", "TTS", f"+{kb(flank_extent)}"]
    for dist in sorted(box_d):
        xticks.append(L["box_right"][np.where(box_d == dist)[0][0]]); xlabels.append(kb(dist))
    ax_c.set_xticks(xticks); ax_c.set_xticklabels(xlabels, fontsize=8, rotation=40, ha="right")
    box_note = (f"; far-field boxes at {' / '.join(kb(d) for d in box_d)}" if nbox else "")
    ax_c.set_xlabel("Distance from gene edge: 5' / upstream (left), 3' / downstream (right). "
                    f"{kb(win)} windows to ±{kb(flank_extent)}{box_note}")
    ax.set_xlim(-0.7, L["xmax"] + 0.7)

    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
