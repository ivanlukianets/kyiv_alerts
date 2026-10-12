import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import kyiv_digital as kd  # noqa: E402
from kyiv_digital import load, parse_duration_minutes  # noqa: E402


def test_durations():
    assert parse_duration_minutes(" 1 годину 17 хвилин ") == 77
    assert parse_duration_minutes(" 31 хвилину ") == 31
    assert parse_duration_minutes(" 2 години ") == 120
    assert parse_duration_minutes(" 6 секунд ") == 0.1
    assert parse_duration_minutes(None) is None


def test_sample():
    a, an, _ = load(HERE / "sample_excerpt.html")
    assert len(a) == 5 and an == {"orphan_end": 0, "double_start": 0, "open_alert": 0, "duration_mismatch": 0}
    assert a.started_at.is_monotonic_increasing
    # 12:51 Kyiv time on 11 Oct 2026 is UTC+3
    assert a.started_at.iloc[0] == pd.Timestamp("2026-10-11 09:51", tz="UTC")
    assert list(a.duration_min.round()) == [11, 31, 78, 70, 14]


def test_dst():
    a, an, _ = load(HERE / "dst_case.html")
    # first alert: 03:40 EEST (00:40 UTC) -> 03:10 EET (01:10 UTC) = 30 min
    # second alert: 03:40 EET (01:40 UTC) -> 04:30 EET (02:30 UTC) = 50 min
    assert list(a.duration_min.round()) == [30, 50], a
    assert an["duration_mismatch"] == 0


def test_odd():
    a, an, _ = load(HERE / "odd_case.html")
    assert an["orphan_end"] == 1 and an["open_alert"] == 1 and len(a) == 2
    assert a.finished_at.isna().sum() == 1


def test_check_does_not_blame_a_running_alert():
    # odd_case.html ends with an alert that is still running; that must not count as a broken record
    res = kd.check_results((HERE / "odd_case.html").read_text(encoding="utf-8"))
    broken = [r for r in res if r[2].startswith("alerts without an all-clear")][0]
    assert broken[1] is True, broken


def test_reject_non_alert_page():
    for bad in ("<html><body>Service unavailable</body></html>", ""):
        try:
            kd.validate_page(bad)
        except ValueError:
            continue
        raise AssertionError("a non-alert page was accepted")
    kd.validate_page((HERE / "sample_excerpt.html").read_text(encoding="utf-8"))  # must not raise


def test_resolve_snapshot_picks_newest():
    sample = (HERE / "sample_excerpt.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        names = ["stats_20261011T000000Z.html", "stats_20261013T080000Z.html", "stats_20261012T105432Z.html"]
        for n in names:
            (d / n).write_text(sample, encoding="utf-8")
        assert kd.resolve_snapshot([d]).name == "stats_20261013T080000Z.html"                  # folder
        assert kd.resolve_snapshot([d / n for n in names]).name == "stats_20261013T080000Z.html"  # list of files
        (d / "stats_20261012T105432Z.html.part").write_text("partial", encoding="utf-8")        # never picked
        assert kd.resolve_snapshot([d]).name == "stats_20261013T080000Z.html"
    with tempfile.TemporaryDirectory() as empty:
        try:
            kd.resolve_snapshot([empty])
        except FileNotFoundError:
            return
        raise AssertionError("empty folder should raise FileNotFoundError")


def test_cli_accepts_several_snapshots():
    """Regression: 'parse snapshots/stats_*.html' used to fail once two snapshots existed."""
    sample = (HERE / "sample_excerpt.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        f1, f2 = d / "stats_20261012T100000Z.html", d / "stats_20261012T110000Z.html"
        f1.write_text(sample, encoding="utf-8")
        f2.write_text(sample, encoding="utf-8")
        out = d / "out.csv"
        r = subprocess.run([sys.executable, str(ROOT / "kyiv_digital.py"), "parse", str(f1), str(f2), "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "using the newest: stats_20261012T110000Z.html" in r.stderr
        df = pd.read_csv(out)
        assert len(df) == 5 and set(df["snapshot_utc"]) == {"2026-10-12T11:00:00Z"}
        # no argument at all, empty snapshots folder -> clear error, not a traceback
        r = subprocess.run([sys.executable, str(ROOT / "kyiv_digital.py"), "check", str(d / "nope")],
                           capture_output=True, text=True)
        assert r.returncode == 2 and "does not exist" in r.stderr


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
