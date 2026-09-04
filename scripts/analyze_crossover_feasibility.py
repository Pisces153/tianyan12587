"""Feasibility calculation for the closed-loop crossover measurement (direction A).

Reads the frozen v3 corpus, estimates the drift diffusion coefficient D from the
debiased structure function of the readout-probe observable, and back-solves the
optimal outer-loop update interval

    sigma_res^2(T) = p(1-p)/(R*T) + (D/2)*T
    T* = sqrt( 2*p(1-p) / (R*D) )
    sigma_min^2 = sqrt( 2*p(1-p)*D / R )

The single question this script answers: does T* fall inside the window a
three-day campaign can actually resolve?

Shot noise is analytically known (binomial), so the structure function is
debiased exactly rather than estimated:

    SF_raw(tau)  = E[(x_j - x_i)^2] = SF_drift(tau) + sigma_i^2 + sigma_j^2
    SF_deb(tau)  = SF_raw(tau) - (sigma_i^2 + sigma_j^2)

No thresholds, no event detection, no scenario labels enter any code path here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

CORPUS = r"E:\TianYan\XA-202609\artifacts\hardware\xa202609_tianyan_287_readout_natural_drift_v3"
PROBE_LABELS = ("readout_all_zero", "readout_all_one")
PROBE_SUCCESS_INDEX = {"readout_all_zero": 0, "readout_all_one": 63}


@dataclass(frozen=True)
class Series:
    t: np.ndarray  # seconds since first submission
    x: np.ndarray  # observable value
    var: np.ndarray  # analytic shot-noise variance of each estimate
    shots: int
    name: str


def load_probe_series(corpus: str) -> dict[str, Series]:
    path = os.path.join(corpus, "snapshots.jsonl")
    recs = [json.loads(line) for line in io.open(path, encoding="utf-8")]
    submitted = {r["snapshot_id"]: r for r in recs if r["event"] == "submitted"}
    collected = [r for r in recs if r["event"] == "collected"]

    times: list[float] = []
    per_label: dict[str, list[float]] = {label: [] for label in PROBE_LABELS}
    shots_seen: set[int] = set()

    for rec in collected:
        sid = rec["snapshot_id"]
        npz = os.path.join(corpus, "raw", f"{sid}_counts.npz")
        if not os.path.exists(npz):
            continue
        blob = np.load(npz, allow_pickle=True)
        labels = list(blob["labels"])
        counts = blob["counts"]
        shots = int(blob["shots"])
        shots_seen.add(shots)
        stamp = submitted.get(sid, rec).get("wallclock_submit_utc", rec["recorded_at_utc"])
        times.append(datetime.fromisoformat(stamp).timestamp())
        for label in PROBE_LABELS:
            row = labels.index(label)
            success = counts[row, PROBE_SUCCESS_INDEX[label]] / shots
            per_label[label].append(1.0 - float(success))

    if len(shots_seen) != 1:
        raise ValueError(f"mixed shot counts in corpus: {sorted(shots_seen)}")
    shots = shots_seen.pop()

    order = np.argsort(np.asarray(times))
    t = np.asarray(times, dtype=float)[order]
    t -= t[0]

    out: dict[str, Series] = {}
    for label in PROBE_LABELS:
        x = np.asarray(per_label[label], dtype=float)[order]
        out[label] = Series(t=t, x=x, var=x * (1.0 - x) / shots, shots=shots, name=label)

    mean_x = np.mean([out[label].x for label in PROBE_LABELS], axis=0)
    mean_var = np.sum([out[label].var for label in PROBE_LABELS], axis=0) / 4.0
    out["readout_mean_error"] = Series(t=t, x=mean_x, var=mean_var, shots=shots, name="readout_mean_error")
    return out


def structure_function(series: Series, edges: np.ndarray) -> list[dict[str, float]]:
    """Debiased pair-wise structure function, binned in lag."""
    n = series.t.size
    i, j = np.triu_indices(n, k=1)
    lag = series.t[j] - series.t[i]
    sq = (series.x[j] - series.x[i]) ** 2
    noise = series.var[i] + series.var[j]

    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lag >= lo) & (lag < hi)
        m = int(sel.sum())
        if m < 8:
            continue
        raw = float(sq[sel].mean())
        floor = float(noise[sel].mean())
        # Pairs share points, so m is not the independent-sample count. Use the
        # number of distinct points involved as a conservative effective n.
        eff = float(np.unique(np.concatenate([i[sel], j[sel]])).size)
        rows.append(
            {
                "lag_lo_s": float(lo),
                "lag_hi_s": float(hi),
                "lag_mid_s": float(lag[sel].mean()),
                "n_pairs": m,
                "n_eff_points": eff,
                "sf_raw": raw,
                "shot_floor": floor,
                "sf_debiased": raw - floor,
                # sampling error of the mean of squares, conservative
                "sf_se": float(sq[sel].std(ddof=1) / np.sqrt(max(eff - 1.0, 1.0))),
            }
        )
    return rows


def fit_power_law(rows: list[dict[str, float]]) -> dict[str, float]:
    pos = [r for r in rows if r["sf_debiased"] > 0]
    if len(pos) < 3:
        return {"ok": 0.0, "n_used": float(len(pos))}
    lx = np.log(np.array([r["lag_mid_s"] for r in pos]))
    ly = np.log(np.array([r["sf_debiased"] for r in pos]))
    alpha, log_c = np.polyfit(lx, ly, 1)
    resid = ly - (alpha * lx + log_c)
    dof = max(len(pos) - 2, 1)
    s2 = float(resid @ resid) / dof
    se_alpha = float(np.sqrt(s2 / np.sum((lx - lx.mean()) ** 2)))
    return {
        "ok": 1.0,
        "n_used": float(len(pos)),
        "alpha": float(alpha),
        "se_alpha": se_alpha,
        "coeff": float(np.exp(log_c)),
        "resid_rms": float(np.sqrt(s2)),
    }


def linear_D(rows: list[dict[str, float]]) -> dict[str, float]:
    """Force alpha = 1 (random walk) and fit SF_deb = D * tau through origin."""
    pos = [r for r in rows if r["sf_debiased"] > 0]
    if not pos:
        return {"ok": 0.0}
    tau = np.array([r["lag_mid_s"] for r in pos])
    sf = np.array([r["sf_debiased"] for r in pos])
    D = float(np.sum(tau * sf) / np.sum(tau * tau))
    return {"ok": 1.0, "D_per_s": D, "n_used": float(len(pos))}


REFERENCE_LABELS = tuple(f"interleaved_reference_{k}" for k in range(1, 5))


def _z0z1(counts: np.ndarray) -> np.ndarray:
    idx = np.arange(64, dtype=np.uint8)
    sign = (1.0 - 2.0 * ((idx >> 5) & 1)) * (1.0 - 2.0 * ((idx >> 4) & 1))
    return (counts @ sign) / counts.sum(axis=-1)


def load_reference_replicates(corpus: str) -> dict[str, np.ndarray]:
    """The four interleaved references are the SAME observable at four
    sub-snapshot times. They are the only handle on lags below the campaign
    cadence, and the only way to separate slow drift from per-job overdispersion.
    """
    path = os.path.join(corpus, "snapshots.jsonl")
    recs = [json.loads(line) for line in io.open(path, encoding="utf-8")]
    submitted = {r["snapshot_id"]: r for r in recs if r["event"] == "submitted"}

    stamps: list[float] = []
    blocks: list[np.ndarray] = []
    positions: set[tuple[int, ...]] = set()
    shots_seen: set[int] = set()
    for rec in (r for r in recs if r["event"] == "collected"):
        sid = rec["snapshot_id"]
        npz = os.path.join(corpus, "raw", f"{sid}_counts.npz")
        if not os.path.exists(npz):
            continue
        blob = np.load(npz, allow_pickle=True)
        labels = list(blob["labels"])
        rows = [labels.index(lbl) for lbl in REFERENCE_LABELS]
        blocks.append(blob["counts"][rows])
        positions.add(tuple(int(blob["positions"][r]) for r in rows))
        shots_seen.add(int(blob["shots"]))
        stamp = submitted.get(sid, rec).get("wallclock_submit_utc", rec["recorded_at_utc"])
        stamps.append(datetime.fromisoformat(stamp).timestamp())

    if len(positions) != 1:
        raise ValueError(f"reference positions vary across snapshots: {positions}")
    order = np.argsort(np.asarray(stamps))
    return {
        "t": np.asarray(stamps)[order],
        "counts": np.asarray(blocks)[order],
        "positions": np.asarray(sorted(positions)[0]),
        "shots": np.asarray(shots_seen.pop()),
    }


def variance_decomposition(ref: dict[str, np.ndarray]) -> dict[str, object]:
    """Split excess variance into within-snapshot (seconds) and between-snapshot
    (>= cadence). Per-job overdispersion inflates BOTH; real drift on timescales
    between the two only inflates the second.
    """
    shots = int(ref["shots"])
    z = _z0z1(ref["counts"].astype(float))  # (n, 4)
    shot_var = (1.0 - z**2) / shots  # +-1 observable

    within = z.var(axis=1, ddof=1)
    snapshot_mean = z.mean(axis=1)
    mean_shot_var = shot_var.mean(axis=1) / 4.0

    within_excess = float(within.mean() - shot_var.mean())
    between_excess = float(snapshot_mean.var(ddof=1) - mean_shot_var.mean())

    return {
        "z_grand_mean": float(z.mean()),
        "positions": [int(p) for p in ref["positions"]],
        "within_var": float(within.mean()),
        "within_shot_var": float(shot_var.mean()),
        "within_ratio": float(within.mean() / shot_var.mean()),
        "within_excess_sd": float(np.sqrt(max(within_excess, 0.0))),
        "between_var": float(snapshot_mean.var(ddof=1)),
        "between_shot_var": float(mean_shot_var.mean()),
        "between_ratio": float(snapshot_mean.var(ddof=1) / mean_shot_var.mean()),
        "between_excess_sd": float(np.sqrt(max(between_excess, 0.0))),
        # fraction of the saturated process variance already developed at the
        # sub-snapshot lag; drives the correlation-time bracket
        "developed_fraction_at_subsnapshot_lag": (
            float(within_excess / between_excess) if between_excess > 0 else float("nan")
        ),
    }


def block_split(ref: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Split at the largest acquisition gap and re-derive per block. Any claim
    that does not survive both blocks independently is not reportable.
    """
    t = ref["t"]
    shots = int(ref["shots"])
    z = _z0z1(ref["counts"].astype(float))
    snapshot_mean = z.mean(axis=1)
    mean_shot_var = ((1.0 - z**2) / shots).mean(axis=1) / 4.0

    gaps = np.diff(t)
    cut = int(np.argmax(gaps)) + 1
    out: list[dict[str, object]] = []
    for lo, hi in ((0, cut), (cut, t.size)):
        n = hi - lo
        if n < 8:
            continue
        v = float(snapshot_mean[lo:hi].var(ddof=1))
        f = float(mean_shot_var[lo:hi].mean())
        out.append(
            {
                "slice": [lo, hi],
                "n": n,
                "span_h": float((t[hi - 1] - t[lo]) / 3600.0),
                "mean": float(snapshot_mean[lo:hi].mean()),
                "var": v,
                "shot_var": f,
                "ratio": v / f,
                "chi2": (n - 1) * v / f,
                "chi2_df": n - 1,
            }
        )
    return out


def crossover(p: float, D: float, R: float) -> dict[str, float]:
    """R = shots per second devoted to the probe observable."""
    if D <= 0 or R <= 0:
        return {"ok": 0.0}
    t_star = float(np.sqrt(2.0 * p * (1.0 - p) / (R * D)))
    sigma_min2 = float(np.sqrt(2.0 * p * (1.0 - p) * D / R))
    return {
        "ok": 1.0,
        "T_star_s": t_star,
        "T_star_min": t_star / 60.0,
        "T_star_h": t_star / 3600.0,
        "B_at_T_star": R * t_star,
        "sigma_min": float(np.sqrt(sigma_min2)),
        "sigma_min2": sigma_min2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=None)
    # three-day design: {10,21} unbundled, 16384 shots each, 50 min cadence
    ap.add_argument("--design-shots-per-setting", type=float, default=16384.0)
    ap.add_argument("--design-cadence-s", type=float, default=3000.0)
    args = ap.parse_args()

    series = load_probe_series(args.corpus)
    span = series["readout_mean_error"].t[-1]
    n = series["readout_mean_error"].t.size
    gaps = np.diff(series["readout_mean_error"].t)

    # log-spaced lag bins from the median gap to half the span
    lo = float(np.median(gaps)) * 0.6
    edges = np.geomspace(lo, span / 2.0, 12)

    report: dict[str, object] = {
        "corpus": args.corpus,
        "n_snapshots": int(n),
        "shots_per_setting": int(series["readout_mean_error"].shots),
        "span_s": float(span),
        "span_days": float(span / 86400.0),
        "median_gap_s": float(np.median(gaps)),
        "probes": {},
    }

    for name in ("readout_all_zero", "readout_all_one", "readout_mean_error"):
        s = series[name]
        rows = structure_function(s, edges)
        pl = fit_power_law(rows)
        lin = linear_D(rows)
        p_bar = float(s.x.mean())
        total_var = float(s.x.var(ddof=1))
        mean_shot_var = float(s.var.mean())

        entry: dict[str, object] = {
            "p_mean": p_bar,
            "total_var": total_var,
            "mean_shot_var": mean_shot_var,
            "excess_var": total_var - mean_shot_var,
            "var_ratio": total_var / mean_shot_var,
            "chi2": float((n - 1) * total_var / mean_shot_var),
            "chi2_df": int(n - 1),
            "structure_function": rows,
            "power_law_fit": pl,
            "linear_fit": lin,
        }

        if lin.get("ok") == 1.0:
            # design throughput for THIS observable
            if name == "readout_mean_error":
                # averaging two settings: effective variance uses 4x total shots
                R = 2.0 * args.design_shots_per_setting / args.design_cadence_s
                R_eff = 2.0 * R  # var = (p0q0+p1q1)/(4B) -> behaves like 2B per setting
            else:
                R_eff = args.design_shots_per_setting / args.design_cadence_s
            entry["design_R_shots_per_s"] = R_eff
            entry["crossover_16384"] = crossover(p_bar, lin["D_per_s"], R_eff)
            entry["crossover_1024"] = crossover(
                p_bar, lin["D_per_s"], R_eff * 1024.0 / args.design_shots_per_setting
            )
            entry["observable_window_s"] = [float(np.median(gaps)), float(span / 2.0)]

        report["probes"][name] = entry  # type: ignore[index]

    ref = load_reference_replicates(args.corpus)
    report["reference_variance_decomposition"] = variance_decomposition(ref)
    report["reference_block_split"] = block_split(ref)

    text = json.dumps(report, indent=2)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
