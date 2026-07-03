#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass

import matplotlib

import numpy as np
import pandas as pd


WARNED_CHROM_RENAMES = set()
INTERNAL_GAP_BINS = 5


@dataclass
class MetaplotConfig:
    """Geometry/aggregation parameters shared by the CLI and importing scripts."""

    bin_size: int
    flanking_bp: int
    body_bins: int
    uniform: bool = False
    value_column: int = 4

    def validate(self):
        if self.bin_size <= 0:
            raise SystemExit("bin_size must be > 0")
        if self.flanking_bp < 0:
            raise SystemExit("flanking_bp must be >= 0")
        if self.flanking_bp % self.bin_size != 0:
            raise SystemExit("flanking_bp must be divisible by bin_size")
        if self.body_bins <= 0:
            raise SystemExit("body_bins must be > 0")
        if self.value_column < 4:
            raise SystemExit("value_column must be 4 or greater")
        return self


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a TSS/TTS metaplot from GFF gene annotations and a signal BED."
    )
    parser.add_argument("--gff", required=True, help="Gene annotation GFF/GFF3 file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Signal BED file. Midpoints are assigned to bins; missing or non-numeric column 4 is treated as 1.",
    )
    parser.add_argument(
        "--value-column",
        type=int,
        default=4,
        help="1-based input column to use as signal value. Default: 4.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        required=True,
        help="Bin size in bp.",
    )
    parser.add_argument(
        "--flanking-bp",
        type=int,
        required=True,
        help="Maximum flank size in bp to include upstream of TSS and downstream of TTS.",
    )
    parser.add_argument(
        "--body-bins",
        type=int,
        required=True,
        help="Number of internal bins to plot on each side of the gene body.",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Distribute each interval's value uniformly across its full span instead of assigning it to its midpoint.",
    )
    parser.add_argument(
        "--output",
        default="metaplot.pdf",
        help="Output PDF path. Default: metaplot.pdf.",
    )
    parser.add_argument(
        "--title",
        default="Metaplot",
        help="Plot title. Default: Metaplot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Write the plot without opening an interactive window.",
    )
    return parser.parse_args(argv)


def normalize_chrom_name(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text.lower().startswith("chr"):
        normalized = f"chr{text[3:]}"
    else:
        normalized = text
    if normalized != text and text not in WARNED_CHROM_RENAMES:
        print(
            f"Warning: normalized chromosome name '{text}' -> '{normalized}'",
            file=sys.stderr,
        )
        WARNED_CHROM_RENAMES.add(text)
    return normalized


def load_genes(path):
    genes = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        names=[
            "chr",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "attributes",
        ],
        usecols=list(range(9)),
    )
    genes = genes[genes["feature"] == "gene"].copy()
    genes["chr"] = genes["chr"].map(normalize_chrom_name)
    genes["start"] = pd.to_numeric(genes["start"], errors="coerce") - 1
    genes["end"] = pd.to_numeric(genes["end"], errors="coerce")
    genes = genes.dropna(subset=["chr", "start", "end", "strand"]).copy()
    genes["start"] = genes["start"].astype(np.int64)
    genes["end"] = genes["end"].astype(np.int64)
    genes["strand"] = genes["strand"].astype(str)
    genes = genes[genes["strand"].isin(["+", "-"])].copy()
    return genes


def load_signal(path, cfg):
    signal = pd.read_csv(path, sep="\t", header=None)
    if signal.shape[1] < 3:
        raise SystemExit(f"Signal BED must have at least 3 columns: {path}")
    if signal.shape[1] >= cfg.value_column:
        value_col = cfg.value_column - 1
        signal = signal.iloc[:, [0, 1, 2, value_col]].copy()
        signal.columns = ["chr", "start", "end", "value"]
    else:
        if cfg.value_column != 4:
            raise SystemExit(
                f"Signal BED has {signal.shape[1]} columns, but value_column {cfg.value_column} was requested"
            )
        signal = signal.iloc[:, :3].copy()
        signal.columns = ["chr", "start", "end"]
        signal["value"] = 1.0

    signal["chr"] = signal["chr"].map(normalize_chrom_name)
    signal["start"] = pd.to_numeric(signal["start"], errors="coerce")
    signal["end"] = pd.to_numeric(signal["end"], errors="coerce")
    signal["value"] = pd.to_numeric(signal["value"], errors="coerce").fillna(1.0)
    signal = signal.dropna(subset=["chr", "start", "end"]).copy()
    signal["start"] = signal["start"].astype(np.int64)
    signal["end"] = signal["end"].astype(np.int64)
    if not cfg.uniform:
        signal["mid"] = ((signal["start"] + signal["end"]) // 2).astype(np.int64)
    return signal


def build_signal_dict(df, cfg):
    signal_dict = {}
    for chrom, sub in df.groupby("chr", sort=False):
        if cfg.uniform:
            starts = sub["start"].to_numpy(dtype=np.int64, copy=True)
            ends = sub["end"].to_numpy(dtype=np.int64, copy=True)
            values = sub["value"].to_numpy(dtype=np.float64, copy=True)
            order = np.argsort(starts, kind="mergesort")
            signal_dict[chrom] = {
                "start": starts[order],
                "end": ends[order],
                "value": values[order],
            }
        else:
            mids = sub["mid"].to_numpy(dtype=np.int64, copy=True)
            values = sub["value"].to_numpy(dtype=np.float64, copy=True)
            order = np.argsort(mids, kind="mergesort")
            signal_dict[chrom] = {
                "mid": mids[order],
                "value": values[order],
            }
    return signal_dict


def midpoint_sum(chrom_signal, window_start, window_end):
    mids = chrom_signal["mid"]
    values = chrom_signal["value"]
    left = np.searchsorted(mids, window_start, side="left")
    right = np.searchsorted(mids, window_end, side="left")
    if right <= left:
        return 0.0
    return float(values[left:right].sum())


def uniform_sum(chrom_signal, window_start, window_end):
    starts = chrom_signal["start"]
    ends = chrom_signal["end"]
    values = chrom_signal["value"]
    right = np.searchsorted(starts, window_end, side="left")
    if right == 0:
        return 0.0
    starts = starts[:right]
    ends = ends[:right]
    values = values[:right]
    valid = ends > window_start
    if not np.any(valid):
        return 0.0
    starts = starts[valid]
    ends = ends[valid]
    values = values[valid]
    overlap = np.minimum(ends, window_end) - np.maximum(starts, window_start)
    positive = overlap > 0
    if not np.any(positive):
        return 0.0
    lengths = np.maximum(1, ends[positive] - starts[positive]).astype(np.float64)
    return float(np.sum(values[positive] * overlap[positive] / lengths))


def add_window(signal_dict, chrom, slot, window_start, window_end, sums, sumsq, counts, cfg):
    chrom_signal = signal_dict.get(chrom)
    if chrom_signal is None:
        return
    else:
        if cfg.uniform:
            value = uniform_sum(chrom_signal, window_start, window_end)
        else:
            value = midpoint_sum(chrom_signal, window_start, window_end)
    sums[slot] += value
    sumsq[slot] += value * value
    counts[slot] += 1


def window_stats(sums, sumsq, counts):
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    variances = np.divide(
        sumsq,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    ) - np.square(np.nan_to_num(means, nan=0.0))
    variances = np.maximum(variances, 0.0)
    ses = np.full_like(sums, np.nan)
    valid = counts > 0
    ses[valid] = np.sqrt(variances[valid] / counts[valid])
    return means, ses


def iter_plus_windows(start, end, bin_size):
    pos = start
    while pos + bin_size <= end:
        yield int(pos), int(pos + bin_size)
        pos += bin_size


def iter_minus_windows(start, end, bin_size):
    pos = end
    while pos - bin_size >= start:
        yield int(pos - bin_size), int(pos)
        pos -= bin_size


def prepare_genes(genes, cfg):
    prepared = []
    for chrom, sub in genes.groupby("chr", sort=False):
        sub = sub.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)
        starts = sub["start"].to_numpy(dtype=np.int64, copy=True)
        ends = sub["end"].to_numpy(dtype=np.int64, copy=True)
        prev_end = np.full(len(sub), -1, dtype=np.int64)
        next_start = np.full(len(sub), np.iinfo(np.int64).max, dtype=np.int64)
        if len(sub) > 1:
            prev_end[1:] = np.maximum.accumulate(ends[:-1])
            next_start[:-1] = starts[1:]

        for idx, row in sub.iterrows():
            strand = row["strand"]
            start = int(row["start"])
            end = int(row["end"])
            if strand == "+":
                tss = start
                tts = end
                flank5_low = max(tss - cfg.flanking_bp, int(prev_end[idx]))
                flank5_high = tss
                flank3_low = tts
                flank3_high = min(tts + cfg.flanking_bp, int(next_start[idx]))
                gene_oriented_start = start
                gene_oriented_end = end
                iter_gene_from_tss = iter_plus_windows
                iter_gene_to_tts = iter_minus_windows
                flank5_reverse = False
                flank3_reverse = False
            else:
                tss = end
                tts = start
                flank5_low = tss
                flank5_high = min(tss + cfg.flanking_bp, int(next_start[idx]))
                flank3_low = max(tts - cfg.flanking_bp, int(prev_end[idx]))
                flank3_high = tts
                gene_oriented_start = start
                gene_oriented_end = end
                iter_gene_from_tss = iter_minus_windows
                iter_gene_to_tts = iter_plus_windows
                flank5_reverse = True
                flank3_reverse = True

            prepared.append(
                {
                    "chr": chrom,
                    "strand": strand,
                    "tss": tss,
                    "tts": tts,
                    "gene_start": gene_oriented_start,
                    "gene_end": gene_oriented_end,
                    "flank5_low": int(flank5_low),
                    "flank5_high": int(flank5_high),
                    "flank3_low": int(flank3_low),
                    "flank3_high": int(flank3_high),
                    "iter_gene_from_tss": iter_gene_from_tss,
                    "iter_gene_to_tts": iter_gene_to_tts,
                    "flank5_reverse": flank5_reverse,
                    "flank3_reverse": flank3_reverse,
                }
            )
    return prepared


def total_slots(cfg):
    flank_bins = cfg.flanking_bp // cfg.bin_size
    return flank_bins + cfg.body_bins + INTERNAL_GAP_BINS + cfg.body_bins + flank_bins


def gene_slot_windows(gene, cfg):
    """Yield (slot, window_start, window_end) for every bin a gene contributes,
    in the flank / TSS-body / gap / TTS-body / flank layout."""
    flank_bins = cfg.flanking_bp // cfg.bin_size
    left_internal_offset = flank_bins
    right_internal_offset = flank_bins + cfg.body_bins + INTERNAL_GAP_BINS
    right_flank_offset = right_internal_offset + cfg.body_bins

    flank5_bins = list(iter_plus_windows(gene["flank5_low"], gene["flank5_high"], cfg.bin_size))
    if gene["flank5_reverse"]:
        flank5_bins.reverse()
    flank5_slots_start = flank_bins - len(flank5_bins)
    for local_idx, (ws, we) in enumerate(flank5_bins):
        yield flank5_slots_start + local_idx, ws, we

    full_gene_bins = list(gene["iter_gene_from_tss"](gene["gene_start"], gene["gene_end"], cfg.bin_size))
    total_gene_bins = len(full_gene_bins)
    if total_gene_bins <= 2 * cfg.body_bins:
        per_side_bins = total_gene_bins // 2
        left_gene_bins = full_gene_bins[:per_side_bins]
        right_gene_bins = list(
            gene["iter_gene_to_tts"](gene["gene_start"], gene["gene_end"], cfg.bin_size)
        )[:per_side_bins]
    else:
        left_gene_bins = full_gene_bins[:cfg.body_bins]
        right_gene_bins = list(
            gene["iter_gene_to_tts"](gene["gene_start"], gene["gene_end"], cfg.bin_size)
        )[:cfg.body_bins]
    right_gene_bins.reverse()

    for local_idx, (ws, we) in enumerate(left_gene_bins):
        yield left_internal_offset + local_idx, ws, we

    right_gene_slots_start = right_flank_offset - len(right_gene_bins)
    for local_idx, (ws, we) in enumerate(right_gene_bins):
        yield right_gene_slots_start + local_idx, ws, we

    flank3_bins = list(iter_plus_windows(gene["flank3_low"], gene["flank3_high"], cfg.bin_size))
    if gene["flank3_reverse"]:
        flank3_bins.reverse()
    for local_idx, (ws, we) in enumerate(flank3_bins):
        yield right_flank_offset + local_idx, ws, we


def _msum(signal_dict, chrom, ws, we):
    cs = signal_dict.get(chrom)
    return 0.0 if cs is None else midpoint_sum(cs, ws, we)


def aggregate_profiles(genes, signal_dict, cfg):
    total_bins = total_slots(cfg)
    sums = np.zeros(total_bins, dtype=np.float64)
    sumsq = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    for gene in genes:
        chrom = gene["chr"]
        for slot, ws, we in gene_slot_windows(gene, cfg):
            add_window(signal_dict, chrom, slot, ws, we, sums, sumsq, counts, cfg)
    averages, ses = window_stats(sums, sumsq, counts)
    return averages, ses, counts


def aggregate_ratio_profile(genes, num_dict, den_dict, cfg):
    """Per-gene pi, averaged across genes that have data.

    For each gene and bin, pi = sum(numerator) / sum(denominator) over that bin's
    windows (numerator = sum_pairwise_differences, denominator = n_sites). Genes
    with 0 sites in a bin are skipped (not counted as zero). The bin value is the
    mean of per-gene pi over contributing genes; counts = number of such genes.
    """
    total_bins = total_slots(cfg)
    sums = np.zeros(total_bins, dtype=np.float64)
    sumsq = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    for gene in genes:
        chrom = gene["chr"]
        gnum: dict[int, float] = {}
        gden: dict[int, float] = {}
        for slot, ws, we in gene_slot_windows(gene, cfg):
            d = _msum(den_dict, chrom, ws, we)
            if d <= 0:
                continue
            gnum[slot] = gnum.get(slot, 0.0) + _msum(num_dict, chrom, ws, we)
            gden[slot] = gden.get(slot, 0.0) + d
        for slot, d in gden.items():
            pi = gnum[slot] / d
            sums[slot] += pi
            sumsq[slot] += pi * pi
            counts[slot] += 1
    averages, ses = window_stats(sums, sumsq, counts)
    return averages, ses, counts


def compute_profile(gff_path, signal_path, cfg):
    """End-to-end: load genes + signal, return (averages, ses, counts)."""
    cfg.validate()
    genes = load_genes(gff_path)
    signal = load_signal(signal_path, cfg)
    gene_layout = prepare_genes(genes, cfg)
    signal_dict = build_signal_dict(signal, cfg)
    return aggregate_profiles(gene_layout, signal_dict, cfg)


def xaxis_ticks(cfg, total_bins):
    """Tick positions/labels for a metaplot built with this config."""
    flank_bins = cfg.flanking_bp // cfg.bin_size
    right_internal_start = flank_bins + cfg.body_bins + INTERNAL_GAP_BINS
    right_flank_start = right_internal_start + cfg.body_bins
    flank_kb = cfg.flanking_bp / 1000.0

    def kb_label(value):
        if float(value).is_integer():
            return f"{int(value)} kb"
        return f"{value:g} kb"

    quarter_flank = flank_bins // 4
    half_flank = flank_bins // 2
    three_quarter_flank = (3 * flank_bins) // 4
    downstream_len = flank_bins
    downstream_quarter = right_flank_start + (downstream_len // 4)
    downstream_half = right_flank_start + (downstream_len // 2)
    downstream_three_quarter = right_flank_start + ((3 * downstream_len) // 4)

    xticks = [
        0,
        quarter_flank,
        half_flank,
        three_quarter_flank,
        flank_bins,
        right_flank_start,
        downstream_quarter,
        downstream_half,
        downstream_three_quarter,
        total_bins - 1,
    ]
    xlabels = [
        f"-{kb_label(flank_kb)}",
        f"-{kb_label(flank_kb * 0.75)}",
        f"-{kb_label(flank_kb * 0.5)}",
        f"-{kb_label(flank_kb * 0.25)}",
        "TSS",
        "TTS",
        f"+{kb_label(flank_kb * 0.25)}",
        f"+{kb_label(flank_kb * 0.5)}",
        f"+{kb_label(flank_kb * 0.75)}",
        f"+{kb_label(flank_kb)}",
    ]
    dedup_ticks = []
    dedup_labels = []
    for tick, label in zip(xticks, xlabels):
        if dedup_ticks and tick == dedup_ticks[-1]:
            continue
        dedup_ticks.append(tick)
        dedup_labels.append(label)
    return dedup_ticks, dedup_labels


def gene_landmarks(cfg):
    """Bin positions of TSS/TTS lines and the gene-body gap span."""
    flank_bins = cfg.flanking_bp // cfg.bin_size
    gap_start = flank_bins + cfg.body_bins
    gap_end = gap_start + INTERNAL_GAP_BINS
    right_flank_start = gap_end + cfg.body_bins
    return {
        "tss": flank_bins,
        "tts": right_flank_start,
        "gap_start": gap_start,
        "gap_end": gap_end,
    }


def build_plot(averages, ses, counts, cfg, title):
    import matplotlib.pyplot as plt

    total_bins = len(averages)
    x = np.arange(total_bins, dtype=np.int64)
    marks = gene_landmarks(cfg)
    gap_start, gap_end = marks["gap_start"], marks["gap_end"]

    fig, (ax, ax_count) = plt.subplots(
        2,
        1,
        figsize=(8, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    ax.fill_between(x, averages - ses, averages + ses, alpha=0.25, linewidth=0)
    ax.plot(x, averages, linewidth=1.5)
    ax.axvline(marks["tss"] - 0.5, linestyle="--", linewidth=1)
    ax.axvline(marks["tts"] - 0.5, linestyle="--", linewidth=1)
    ax.axvspan(gap_start - 0.5, gap_end - 0.5, color="#d9d9d9", alpha=0.6, zorder=0)
    ax.axvline(gap_start - 0.5, linestyle=":", linewidth=1, color="gray")
    ax.axvline(gap_end - 0.5, linestyle=":", linewidth=1, color="gray")
    ax.set_ylabel("Average signal")
    ax.set_title(title)

    ax_count.bar(x, counts, width=1.0, color="#8da0cb", edgecolor="none")
    ax_count.axvline(marks["tss"] - 0.5, linestyle="--", linewidth=1)
    ax_count.axvline(marks["tts"] - 0.5, linestyle="--", linewidth=1)
    ax_count.axvspan(gap_start - 0.5, gap_end - 0.5, color="#d9d9d9", alpha=0.6, zorder=0)
    ax_count.axvline(gap_start - 0.5, linestyle=":", linewidth=1, color="gray")
    ax_count.axvline(gap_end - 0.5, linestyle=":", linewidth=1, color="gray")
    ax_count.set_ylabel("Genes\ncontributing")

    internal_bp = 2 * cfg.body_bins * cfg.bin_size
    internal_kb = internal_bp / 1000.0
    internal_label = f"{int(internal_kb)} kb" if float(internal_kb).is_integer() else f"{internal_kb:g} kb"

    dedup_ticks, dedup_labels = xaxis_ticks(cfg, total_bins)
    ax_count.set_xticks(dedup_ticks)
    ax_count.set_xticklabels(dedup_labels)
    ax_count.set_xlabel(
        f"Flank / gene windows ({cfg.body_bins} bins from TSS, gap, {cfg.body_bins} bins from TTS; {internal_label} total shown inside gene)"
    )
    ax.set_xlim(0, total_bins - 1)
    ax.tick_params(axis="x", labelbottom=False)
    plt.tight_layout()
    return fig


def main(argv=None):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    cfg = MetaplotConfig(
        bin_size=args.bin_size,
        flanking_bp=args.flanking_bp,
        body_bins=args.body_bins,
        uniform=args.uniform,
        value_column=args.value_column,
    ).validate()

    averages, ses, counts = compute_profile(args.gff, args.input, cfg)
    fig = build_plot(averages, ses, counts, cfg, args.title)
    fig.savefig(args.output, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
