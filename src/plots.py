from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


RAW_DIR = Path("results/raw")
FIG_DIR = Path("results/figures")


PROTOCOL_ORDER = ["Raft", "Gossip", "CRDT"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_results(
    summary_path: Path = RAW_DIR / "summary.csv",
    time_series_path: Path = RAW_DIR / "time_series.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    if not time_series_path.exists():
        raise FileNotFoundError(f"Missing time-series CSV: {time_series_path}")

    summary = pd.read_csv(summary_path)
    time_series = pd.read_csv(time_series_path)

    return summary, time_series


def ordered_protocols(values: Iterable[str]) -> list[str]:
    present = set(values)
    ordered = [p for p in PROTOCOL_ORDER if p in present]
    ordered.extend(sorted(p for p in present if p not in PROTOCOL_ORDER))
    return ordered


def median_iqr(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    grouped = (
        df
        .groupby(group_cols, dropna=False)[value_col]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            count="count",
        )
        .reset_index()
    )
    return grouped


def save_current_figure(stem: str) -> None:
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_fig1_partition_availability(summary: pd.DataFrame) -> None:
    """
    Figure 1:
    Immediate availability for proposals issued during the partition epoch.

    For Raft, this is immediate commit availability for the sampled epoch.
    For Gossip and CRDT, this is local actuation availability for the sampled epoch.
    """
    df = summary[summary["experiment"] == "E1_partition_progress"].copy()

    if df.empty:
        print("Skipping Fig. 1: no E1 data")
        return

    metric = "epoch_partition_progress_availability"

    if metric not in df.columns:
        print(f"Skipping Fig. 1: missing {metric}")
        return

    agg = median_iqr(df, ["protocol"], metric)
    protocols = ordered_protocols(agg["protocol"].tolist())
    agg = agg.set_index("protocol").loc[protocols].reset_index()

    yerr_lower = agg["median"] - agg["q1"]
    yerr_upper = agg["q3"] - agg["median"]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    ax.bar(
        agg["protocol"],
        agg["median"],
        yerr=[yerr_lower, yerr_upper],
        capsize=4,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Partition-window availability")
    ax.set_xlabel("Protocol")
    ax.set_title("Immediate availability during partition")

    ax.text(
        0.5,
        -0.24,
        "Raft reports immediate commit availability; Gossip and CRDT report local actuation availability.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    save_current_figure("fig1_partition_availability")


def plot_fig2_loss_latency(summary: pd.DataFrame) -> None:
    """
    Figure 2:
    Latency under packet loss.

    Panel A: local actuation latency.
    Panel B: convergence latency.
    """
    df = summary[summary["experiment"] == "E2_loss_vs_latency"].copy()

    if df.empty:
        print("Skipping Fig. 2: no E2 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)

    metrics = [
        ("mean_L_act_ms", "Local actuation latency $L_{act}$ (ms)"),
        ("L_conv_ms", "Convergence latency $L_{conv}$ (ms)"),
    ]

    protocols = ordered_protocols(df["protocol"].unique())

    for ax, (metric, ylabel) in zip(axes, metrics):
        for protocol in protocols:
            sub = df[df["protocol"] == protocol].copy()
            sub = sub.dropna(subset=["packet_loss", metric])

            if sub.empty:
                continue

            agg = median_iqr(sub, ["packet_loss"], metric)
            agg = agg.sort_values("packet_loss")

            ax.plot(
                agg["packet_loss"],
                agg["median"],
                marker="o",
                label=protocol,
            )

        ax.set_xlabel("Packet-loss probability")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Actuation latency")
    axes[1].set_title("Convergence latency")
    axes[1].legend(title="Protocol", loc="best")

    save_current_figure("fig2_loss_latency")


def plot_fig3_healing_after_reconnect(time_series: pd.DataFrame) -> None:
    """
    Figure 3:
    Healing after reconnect.

    Uses convergence gap = 1 - convergence_fraction.

    This intentionally includes unresolved states. It should not be interpreted
    as pure decision disagreement among already-decided robots.
    """
    df = time_series[time_series["experiment"] == "E3_healing_after_reconnect"].copy()

    if df.empty:
        print("Skipping Fig. 3: no E3 time-series data")
        return

    if "partition_end_ms" not in df.columns:
        print("Skipping Fig. 3: partition_end_ms missing")
        return

    df = df.dropna(subset=["partition_end_ms"])
    df["time_after_reconnect_s"] = (df["time_ms"] - df["partition_end_ms"]) / 1000.0
    df = df[df["time_after_reconnect_s"] >= -2.0].copy()

    df["convergence_gap"] = 1.0 - df["convergence_fraction"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    protocols = ordered_protocols(df["protocol"].unique())

    for protocol in protocols:
        sub = df[df["protocol"] == protocol].copy()

        if sub.empty:
            continue

        agg = median_iqr(
            sub,
            ["time_after_reconnect_s"],
            "convergence_gap",
        )
        agg = agg.sort_values("time_after_reconnect_s")

        ax.plot(
            agg["time_after_reconnect_s"],
            agg["median"],
            label=protocol,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time after reconnect (s)")
    ax.set_ylabel("Convergence gap")
    ax.set_title("Healing after partition reconnect")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Protocol", loc="best")

    ax.text(
        0.5,
        -0.25,
        "Convergence gap is 1 minus the dominant non-null decision fraction; it includes unresolved robots.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    save_current_figure("fig3_healing_after_reconnect")


def plot_fig4_scalability_overhead(summary: pd.DataFrame) -> None:
    """
    Figure 4:
    Scalability and overhead.

    Panel A: transmitted bytes per robot.
    Panel B: active coordination-state items per robot.
    """
    df = summary[summary["experiment"] == "E4_scalability_overhead"].copy()

    if df.empty:
        print("Skipping Fig. 4: no E4 data")
        return

    df["bytes_per_robot"] = df["bytes_sent"] / df["n"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)

    metrics = [
        ("bytes_per_robot", "Bytes sent per robot"),
        ("active_state_items_mean", "Active state items per robot"),
    ]

    protocols = ordered_protocols(df["protocol"].unique())

    for ax, (metric, ylabel) in zip(axes, metrics):
        for protocol in protocols:
            sub = df[df["protocol"] == protocol].copy()
            sub = sub.dropna(subset=["n", metric])

            if sub.empty:
                continue

            agg = median_iqr(sub, ["n"], metric)
            agg = agg.sort_values("n")

            ax.plot(
                agg["n"],
                agg["median"],
                marker="o",
                label=protocol,
            )

        ax.set_xlabel("Swarm size $N$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Overhead spans orders of magnitude; log scale makes the comparison readable.
        ax.set_yscale("log")

    axes[0].set_title("Network overhead")
    axes[1].set_title("State-size overhead")
    axes[1].legend(title="Protocol", loc="best")

    save_current_figure("fig4_scalability_overhead")


def write_manifest(summary: pd.DataFrame, time_series: pd.DataFrame) -> None:
    manifest = {
        "git_commit": git_commit_hash(),
        "summary_rows": int(len(summary)),
        "time_series_rows": int(len(time_series)),
        "experiments": sorted(summary["experiment"].dropna().unique().tolist()),
        "protocols": sorted(summary["protocol"].dropna().unique().tolist()),
        "seed_count_by_experiment_protocol": (
            summary
            .groupby(["experiment", "protocol"])["seed"]
            .nunique()
            .reset_index(name="seed_count")
            .to_dict(orient="records")
        ),
        "figures": [
            "fig1_partition_availability.png",
            "fig1_partition_availability.pdf",
            "fig2_loss_latency.png",
            "fig2_loss_latency.pdf",
            "fig3_healing_after_reconnect.png",
            "fig3_healing_after_reconnect.pdf",
            "fig4_scalability_overhead.png",
            "fig4_scalability_overhead.pdf",
        ],
    }

    path = FIG_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {path}")


def generate_all_plots() -> None:
    ensure_dirs()

    summary, time_series = load_results()

    plot_fig1_partition_availability(summary)
    plot_fig2_loss_latency(summary)
    plot_fig3_healing_after_reconnect(time_series)
    plot_fig4_scalability_overhead(summary)
    write_manifest(summary, time_series)