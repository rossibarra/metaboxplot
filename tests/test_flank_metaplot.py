import re
import sys

import numpy as np
import pandas as pd
import pytest

import flank_metaplot as fm


def _track(rows, *, col=4, kind="value"):
    df = pd.DataFrame(rows, columns=["chr", "start", "end", 3])
    return fm.build_track(df, col, kind)["chroms"]["chr1"]


def test_region_vals_value_mode_counts_gaps_as_zero_and_skips_empty_windows():
    chrom = _track(
        [
            ("chr1", 0, 10, 2.0),
            ("chr1", 20, 30, 4.0),
        ],
        kind="value",
    )
    track = {"names": ["cov", "val"]}

    vals, ok = fm.region_vals(
        chrom,
        lo=np.array([0, 5, 10, 15, 20, 5]),
        hi=np.array([10, 15, 20, 25, 30, 5]),
        track=track,
    )

    np.testing.assert_allclose(vals[:5], [2.0, 1.0, 0.0, 2.0, 4.0])
    assert ok.tolist() == [True, True, False, True, True, False]
    assert np.isnan(vals[5])


def test_region_vals_event_mode_returns_per_bp_density_over_full_window_width():
    chrom = _track(
        [
            ("chr1", 0, 10, 5.0),
            ("chr1", 20, 30, 10.0),
        ],
        kind="event",
    )
    track = {"names": ["cov", "val"]}

    vals, ok = fm.region_vals(
        chrom,
        lo=np.array([0, 0, 10, 0, 20]),
        hi=np.array([10, 20, 20, 30, 30]),
        track=track,
    )

    np.testing.assert_allclose(vals, [0.5, 0.25, 0.0, 0.5, 1.0])
    assert ok.tolist() == [True, True, False, True, True]


def test_region_vals_uses_overlap_weighted_integrals_for_variable_width_intervals():
    value_chrom = _track(
        [
            ("chr1", 0, 3, 6.0),
            ("chr1", 3, 9, 3.0),
        ],
        kind="value",
    )
    event_chrom = _track(
        [
            ("chr1", 0, 3, 6.0),
            ("chr1", 3, 9, 3.0),
        ],
        kind="event",
    )
    track = {"names": ["cov", "val"]}

    value_vals, value_ok = fm.region_vals(value_chrom, [0], [9], track)
    event_vals, event_ok = fm.region_vals(event_chrom, [0], [9], track)

    np.testing.assert_allclose(value_vals, [4.0])
    np.testing.assert_allclose(event_vals, [1.0])
    assert value_ok.tolist() == [True]
    assert event_ok.tolist() == [True]


def test_build_track_rejects_overlapping_ranges():
    df = pd.DataFrame(
        [
            ("chr1", 0, 10, 1.0),
            ("chr1", 9, 20, 2.0),
        ],
        columns=["chr", "start", "end", 3],
    )

    with pytest.raises(SystemExit, match=r"overlapping ranges.*chr1"):
        fm.build_track(df, 4, "value")


def test_load_bed_df_reads_only_requested_value_columns(tmp_path):
    bed = tmp_path / "signal.bedGraph"
    bed.write_text("track type=bedGraph\nchr1\t0\t10\t1\t2\t3\n")

    df = fm.load_bed_df(bed, {6})

    assert list(df.columns) == ["chr", "start", "end", 5]
    assert df.loc[0, 5] == 3


def test_load_bed_df_reports_missing_requested_column(tmp_path):
    bed = tmp_path / "signal.bedGraph"
    bed.write_text("chr1\t0\t10\t1\n")

    with pytest.raises(SystemExit, match=r"column\(s\) 6 not present"):
        fm.load_bed_df(bed, {6})


def test_series_action_binds_columns_to_most_recent_bed_in_command_order(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flank_metaplot.py",
            "--gff",
            "genes.gff3",
            "--bed",
            "first.bedGraph",
            "--event",
            "4,5",
            "--bed",
            "second.bedGraph",
            "--value",
            "6",
            "--output",
            "out.pdf",
        ],
    )

    args = fm.parse_args()

    assert args.series == [
        {"kind": "event", "col": 4, "bed": 0},
        {"kind": "event", "col": 5, "bed": 0},
        {"kind": "value", "col": 6, "bed": 1},
    ]


def test_main_accumulates_plus_and_minus_strands_in_five_prime_to_three_prime_orientation(
    tmp_path, monkeypatch
):
    gff = tmp_path / "genes.gff3"
    gff.write_text(
        "\n".join(
            [
                "chr1\t.\tgene\t101\t200\t.\t+\t.\tID=plus",
                "chr1\t.\tgene\t301\t400\t.\t-\t.\tID=minus",
            ]
        )
        + "\n"
    )
    bed = tmp_path / "signal.bedGraph"
    bed.write_text(
        "\n".join(
            [
                "chr1\t80\t90\t12",
                "chr1\t90\t100\t11",
                "chr1\t100\t110\t31",
                "chr1\t110\t120\t32",
                "chr1\t180\t190\t42",
                "chr1\t190\t200\t41",
                "chr1\t200\t210\t21",
                "chr1\t210\t220\t22",
                "chr1\t280\t290\t62",
                "chr1\t290\t300\t61",
                "chr1\t300\t310\t81",
                "chr1\t310\t320\t82",
                "chr1\t380\t390\t72",
                "chr1\t390\t400\t71",
                "chr1\t400\t410\t51",
                "chr1\t410\t420\t52",
            ]
        )
        + "\n"
    )
    calls = []
    original_acc_add = fm.acc_add

    def spy_acc_add(acc, idx, val, ok):
        calls.append((np.asarray(idx).copy(), np.asarray(val).copy(), np.asarray(ok).copy()))
        return original_acc_add(acc, idx, val, ok)

    monkeypatch.setattr(fm, "acc_add", spy_acc_add)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flank_metaplot.py",
            "--gff",
            str(gff),
            "--bed",
            str(bed),
            "--value",
            "4",
            "--win",
            "10",
            "--flank-bp",
            "20",
            "--body-bins",
            "2",
            "--box-dists",
            "--output",
            str(tmp_path / "out.pdf"),
        ],
    )

    assert fm.main() == 0

    got = [tuple(call[1].tolist()) for call in calls]
    assert got == [
        (11.0, 12.0),  # plus upstream: nearest 5' edge first, then farther out
        (21.0, 22.0),  # plus downstream from TTS
        (31.0, 32.0),  # plus body inward from TSS
        (41.0, 42.0),  # plus body inward from TTS
        (51.0, 52.0),  # minus upstream is right of the genomic end
        (61.0, 62.0),  # minus downstream is left of the genomic start
        (71.0, 72.0),  # minus TSS body bins move left from genomic end
        (81.0, 82.0),  # minus TTS body bins move right from genomic start
    ]
    assert all(call[2].tolist() == [True, True] for call in calls)


@pytest.mark.parametrize(
    ("option", "bad_value", "message"),
    [
        ("--win", "0", "win"),
        ("--flank-bp", "-1", "flank"),
        ("--flank-bp", "499", "at least --win"),
        ("--body-bins", "-1", "body"),
        ("--box-halfwidth", "-1", "box"),
        ("--box-halfwidth", "0", "box-halfwidth"),
        ("--box-dists", "-1", "box-dists"),
    ],
)
def test_main_reports_helpful_errors_for_invalid_geometry_options(
    tmp_path, monkeypatch, option, bad_value, message
):
    gff = tmp_path / "genes.gff3"
    gff.write_text("chr1\t.\tgene\t101\t200\t.\t+\t.\tID=g1\n")
    bed = tmp_path / "signal.bedGraph"
    bed.write_text("chr1\t0\t300\t1\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flank_metaplot.py",
            "--gff",
            str(gff),
            "--bed",
            str(bed),
            "--value",
            "4",
            option,
            bad_value,
            "--output",
            str(tmp_path / "out.pdf"),
        ],
    )

    with pytest.raises(SystemExit, match=re.compile(message, re.IGNORECASE)):
        fm.main()
