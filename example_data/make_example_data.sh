#!/usr/bin/env bash
# Build the compact example dataset for flank_metaplot.py.
#
# Region: chr10:0-40,000,000 of maize B73 v5 (637 genes). Produces small,
# non-overlapping 500 bp bedGraphs plus a gene GFF subset:
#   genes.chr10_0-40Mb.gff3     gene annotations (GFF3, 'gene' features)
#   recomb_rate.500bp.bedGraph  recombination rate, cM/Mb   -> use as --value
#   pi.500bp.bedGraph           nucleotide diversity, pi    -> use as --value
#   crossovers.500bp.bedGraph   expected crossovers/window  -> use as --event
#
# Sources live in the linked_selection project; edit SRC if they move. Re-running
# regenerates the files deterministically.
set -euo pipefail

SRC=/quobyte/jrigrp/jri/projects/linked_selection
GFF=$SRC/data/v5.genes.gff3
RECOMB=$SRC/results/recomb/finemap_v5.rate.500bp.bed
PI=$SRC/results/diversity/all_chroms.pi.500bp.bed
XO=$SRC/data/xo_Combined_LR13_LR14_parents2_v5.bed

CHR=chr10
START=0
END=40000000
WIN=500
OUT=$(cd "$(dirname "$0")" && pwd)

# --- genes (GFF3): keep 'gene' features on the region ---
awk -F'\t' -v OFS='\t' -v e="$END" '
  !/^#/ && $1=="'"$CHR"'" && $3=="gene" && $4>=1 && $5<=e' "$GFF" \
  > "$OUT/genes.${CHR}_0-40Mb.gff3"

# --- recombination rate (value): source chrom is "Chr10"; relabel to chr10 ---
awk -F'\t' -v OFS='\t' -v s="$START" -v e="$END" '
  ($1=="Chr10"||$1=="chr10") && $2>=s && $3<=e {print "'"$CHR"'",$2,$3,$4}' "$RECOMB" \
  > "$OUT/recomb_rate.500bp.bedGraph"

# --- nucleotide diversity pi (value) ---
awk -F'\t' -v OFS='\t' -v s="$START" -v e="$END" '
  $1=="'"$CHR"'" && $2>=s && $3<=e {print $1,$2,$3,$4}' "$PI" \
  > "$OUT/pi.500bp.bedGraph"

# --- crossovers (event): distribute each crossover segment uniformly over its
#     length into 500 bp windows -> expected # crossovers per window ---
XOTMP="$OUT/.xo_segments.tmp"
awk -F'\t' -v OFS='\t' -v e="$END" '
  ($1=="Chr10"||$1=="chr10") && $2<e && $3>0 {print $2,$3}' "$XO" > "$XOTMP"
# (python3 - reads the program from the heredoc; segment data comes from XOTMP)
python3 - "$XOTMP" "$START" "$END" "$WIN" "$OUT/crossovers.500bp.bedGraph" "$CHR" <<'PY'
import sys
import numpy as np

data = sys.argv[1]
start, end, win = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out, chrom = sys.argv[5], sys.argv[6]
nwin = (end - start) // win
counts = np.zeros(nwin)
with open(data) as fh:
    for line in fh:
        s, e = (int(x) for x in line.split())
        s = max(s, start); e = min(e, end)
        if e <= s:
            continue
        length = e - s                       # 1 crossover spread over the segment
        a, b = (s - start) // win, (e - 1 - start) // win
        wi = np.arange(a, b + 1)
        wlo = start + wi * win
        ov = np.minimum(e, wlo + win) - np.maximum(s, wlo)
        counts[wi] += ov / length            # expected crossovers landing in each window
with open(out, "w") as fh:
    for i, c in enumerate(counts):
        lo = start + i * win
        fh.write(f"{chrom}\t{lo}\t{lo + win}\t{c:.6g}\n")
PY
rm -f "$XOTMP"

echo "Wrote example data to $OUT :"
wc -l "$OUT"/genes.${CHR}_0-40Mb.gff3 "$OUT"/*.bedGraph
