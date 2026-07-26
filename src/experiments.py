from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Type

import pandas as pd

from .model import Proposal
from .protocols import CRDTIntent, GossipOverwrite, ProtocolBase, RaftIntent
from .simulator import Simulator


PROTOCOLS: List[Type[ProtocolBase]] = [
    RaftIntent,
    GossipOverwrite,
    CRDTIntent,
]

ACTIONS = ["a1", "a2"]


def ensure_dirs() -> None:
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def alternating_action(robot: int) -> str:
    return ACTIONS[robot % 2]


def component_action(robot: int, n: int) -> str:
    return "a1" if robot < n // 2 else "a2"


def all_robot_wave(time_ms: int, n: int, epoch: int, mode: str) -> List[Proposal]:
    proposals: List[Proposal] = []

    for robot in range(n):
        if mode == "same":
            action = "a1"
        elif mode == "alternating":
            action = alternating_action(robot)
        elif mode == "component":
            action = component_action(robot, n)
        else:
            raise ValueError(f"Unknown workload mode: {mode}")

        proposals.append(
            Proposal(
                time_ms=time_ms + robot,
                robot=robot,
                epoch=epoch,
                action=action,
            )
        )

    return proposals


def schedule_sampling(
    sim: Simulator,
    rows: List[Dict],
    experiment: str,
    epoch: int,
    start_ms: int,
    end_ms: int,
    every_ms: int,
    extra: Dict,
) -> None:
    for t in range(start_ms, end_ms + 1, every_ms):
        def make_callback(sample_time: int):
            def sample() -> None:
                row = sim.sample_row(experiment=experiment, epoch=epoch)
                row.update(extra)
                row["scheduled_sample_time_ms"] = sample_time
                rows.append(row)
            return sample

        sim.schedule_at(t, make_callback(t))

"""
def first_time_at_agreement(
    rows: List[Dict],
    first_proposal_time_ms: int,
    threshold: float = 0.95,
    after_ms: int | None = None,
) -> float:
    start = first_proposal_time_ms if after_ms is None else after_ms

    ordered = sorted(rows, key=lambda r: r["time_ms"])
    for row in ordered:
        if row["time_ms"] < start:
            continue
        if row["agreement_fraction"] >= threshold:
            return float(row["time_ms"] - first_proposal_time_ms)

    return math.nan

"""

def first_time_at_agreement(
    rows: List[Dict],
    first_proposal_time_ms: int,
    threshold: float = 0.95,
    after_ms: int | None = None,
) -> float:
    start = first_proposal_time_ms if after_ms is None else after_ms

    ordered = sorted(rows, key=lambda r: r["time_ms"])
    for row in ordered:
        if row["time_ms"] < start:
            continue
        if row["convergence_fraction"] >= threshold:
            return float(row["time_ms"] - first_proposal_time_ms)

    return math.nan

"""    
def first_healing_time_after(
    rows: List[Dict],
    partition_end_ms: int,
    threshold: float = 0.95,
) -> float:
    ordered = sorted(rows, key=lambda r: r["time_ms"])
    for row in ordered:
        if row["time_ms"] < partition_end_ms:
            continue
        if row["agreement_fraction"] >= threshold:
            return float(row["time_ms"] - partition_end_ms)

    return math.nan

"""

def first_healing_time_after(
    rows: List[Dict],
    partition_end_ms: int,
    threshold: float = 0.95,
) -> float:
    ordered = sorted(rows, key=lambda r: r["time_ms"])
    for row in ordered:
        if row["time_ms"] < partition_end_ms:
            continue
        if row["convergence_fraction"] >= threshold:
            return float(row["time_ms"] - partition_end_ms)

    return math.nan


def run_one(
    *,
    experiment: str,
    protocol_cls: Type[ProtocolBase],
    n: int,
    seed: int,
    duration_ms: int,
    proposals: List[Proposal],
    sample_epoch: int,
    packet_loss: float,
    gossip_period_ms: int = 50,
    gossip_fanout: int = 2,
    partition_start_ms: int | None = None,
    partition_end_ms: int | None = None,
    sample_every_ms: int = 100,
    extra: Dict | None = None,
) -> tuple[Dict, List[Dict]]:
    extra = extra or {}

    sim = Simulator(
        n=n,
        protocol_cls=protocol_cls,
        duration_ms=duration_ms,
        seed=seed,
        packet_loss=packet_loss,
        gossip_period_ms=gossip_period_ms,
        gossip_fanout=gossip_fanout,
        partition_start_ms=partition_start_ms,
        partition_end_ms=partition_end_ms,
    )

    for proposal in proposals:
        sim.schedule_proposal(proposal)

    time_rows: List[Dict] = []
    schedule_sampling(
        sim=sim,
        rows=time_rows,
        experiment=experiment,
        epoch=sample_epoch,
        start_ms=0,
        end_ms=duration_ms,
        every_ms=sample_every_ms,
        extra={
            **extra,
            "packet_loss": packet_loss,
            "gossip_period_ms": gossip_period_ms,
            "gossip_fanout": gossip_fanout,
            "partition_start_ms": partition_start_ms,
            "partition_end_ms": partition_end_ms,
        },
    )

    sim.run()

    summary = sim.summary(epoch=sample_epoch)
    first_proposal_time = sim.first_proposal_time_by_epoch.get(sample_epoch, 0)

    """
    summary.update(
        {
            "experiment": experiment,
            "packet_loss": packet_loss,
            "gossip_period_ms": gossip_period_ms,
            "gossip_fanout": gossip_fanout,
            "partition_start_ms": partition_start_ms,
            "partition_end_ms": partition_end_ms,
            "sample_epoch": sample_epoch,
            "L_conv_ms": first_time_at_agreement(
                time_rows,
                first_proposal_time_ms=first_proposal_time,
                threshold=0.95,
            ),
            "max_delta_decision": max(
                (r["delta_decision"] for r in time_rows),
                default=math.nan,
            ),
            "mean_delta_decision": (
                sum(r["delta_decision"] for r in time_rows) / len(time_rows)
                if time_rows
                else math.nan
            ),
            "max_active_state_items_mean": max(
                (r["active_state_items_mean"] for r in time_rows),
                default=math.nan,
            ),
            **extra,
        }
    )
    """

    delta_values = [
        float(r["delta_decision"])
        for r in time_rows
        if r["delta_decision"] == r["delta_decision"]  # exclude NaN
    ]

    summary.update(
        {
            "experiment": experiment,
            "packet_loss": packet_loss,
            "gossip_period_ms": gossip_period_ms,
            "gossip_fanout": gossip_fanout,
            "partition_start_ms": partition_start_ms,
            "partition_end_ms": partition_end_ms,
            "sample_epoch": sample_epoch,
            "L_conv_ms": first_time_at_agreement(
                time_rows,
                first_proposal_time_ms=first_proposal_time,
                threshold=0.95,
            ),
            "max_delta_decision": max(delta_values) if delta_values else math.nan,
            "mean_delta_decision": (
                sum(delta_values) / len(delta_values)
                if delta_values
                else math.nan
            ),
            "max_unresolved_fraction": max(
                (r["unresolved_fraction"] for r in time_rows),
                default=math.nan,
            ),
            "max_active_state_items_mean": max(
                (r["active_state_items_mean"] for r in time_rows),
                default=math.nan,
            ),
            **extra,
        }
    )


    if partition_end_ms is not None:
        summary["healing_ms_after_partition"] = first_healing_time_after(
            time_rows,
            partition_end_ms=partition_end_ms,
            threshold=0.95,
        )
    else:
        summary["healing_ms_after_partition"] = math.nan

    return summary, time_rows


def run_e1_partition_progress(seeds: Iterable[int]) -> tuple[List[Dict], List[Dict]]:
    experiment = "E1_partition_progress"
    summaries: List[Dict] = []
    time_rows: List[Dict] = []

    n = 50
    duration_ms = 30000
    partition_start_ms = 10000
    partition_end_ms = 20000

    proposals = []
    proposals += all_robot_wave(2000, n, epoch=1, mode="same")
    proposals += all_robot_wave(12000, n, epoch=2, mode="component")
    proposals += all_robot_wave(24000, n, epoch=3, mode="same")

    for seed in seeds:
        for protocol_cls in PROTOCOLS:
            summary, rows = run_one(
                experiment=experiment,
                protocol_cls=protocol_cls,
                n=n,
                seed=seed,
                duration_ms=duration_ms,
                proposals=proposals,
                sample_epoch=2,
                packet_loss=0.05,
                partition_start_ms=partition_start_ms,
                partition_end_ms=partition_end_ms,
                extra={"scenario": "no_majority_partition"},
            )
            summaries.append(summary)
            time_rows.extend(rows)

    return summaries, time_rows


def run_e2_loss_vs_latency(seeds: Iterable[int]) -> tuple[List[Dict], List[Dict]]:
    experiment = "E2_loss_vs_latency"
    summaries: List[Dict] = []
    time_rows: List[Dict] = []

    n = 50
    duration_ms = 20000
    loss_values = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

    for packet_loss in loss_values:
        proposals = all_robot_wave(2000, n, epoch=1, mode="alternating")

        for seed in seeds:
            for protocol_cls in PROTOCOLS:
                summary, rows = run_one(
                    experiment=experiment,
                    protocol_cls=protocol_cls,
                    n=n,
                    seed=seed,
                    duration_ms=duration_ms,
                    proposals=proposals,
                    sample_epoch=1,
                    packet_loss=packet_loss,
                    partition_start_ms=None,
                    partition_end_ms=None,
                    extra={"scenario": "loss_sweep"},
                )
                summaries.append(summary)
                time_rows.extend(rows)

    return summaries, time_rows


def run_e3_healing_after_reconnect(seeds: Iterable[int]) -> tuple[List[Dict], List[Dict]]:
    experiment = "E3_healing_after_reconnect"
    summaries: List[Dict] = []
    time_rows: List[Dict] = []

    n = 50
    duration_ms = 30000
    partition_start_ms = 5000
    partition_end_ms = 15000

    # Conflicting component-local proposals during the partition.
    proposals = all_robot_wave(7000, n, epoch=1, mode="component")

    for seed in seeds:
        for protocol_cls in PROTOCOLS:
            summary, rows = run_one(
                experiment=experiment,
                protocol_cls=protocol_cls,
                n=n,
                seed=seed,
                duration_ms=duration_ms,
                proposals=proposals,
                sample_epoch=1,
                packet_loss=0.05,
                partition_start_ms=partition_start_ms,
                partition_end_ms=partition_end_ms,
                extra={"scenario": "partition_healing"},
            )
            summaries.append(summary)
            time_rows.extend(rows)

    return summaries, time_rows


def run_e4_scalability_overhead(seeds: Iterable[int], quick: bool = True) -> tuple[List[Dict], List[Dict]]:
    experiment = "E4_scalability_overhead"
    summaries: List[Dict] = []
    time_rows: List[Dict] = []

    n_values = [10, 25, 50, 100] if quick else [10, 25, 50, 100, 250, 500, 1000]
    duration_ms = 15000

    for n in n_values:
        proposals: List[Proposal] = []
        for epoch in range(1, 6):
            proposals += all_robot_wave(
                1000 + epoch * 1000,
                n,
                epoch=epoch,
                mode="alternating",
            )

        for seed in seeds:
            for protocol_cls in PROTOCOLS:
                summary, rows = run_one(
                    experiment=experiment,
                    protocol_cls=protocol_cls,
                    n=n,
                    seed=seed,
                    duration_ms=duration_ms,
                    proposals=proposals,
                    sample_epoch=5,
                    packet_loss=0.05,
                    partition_start_ms=None,
                    partition_end_ms=None,
                    sample_every_ms=250,
                    extra={"scenario": "n_sweep"},
                )
                summaries.append(summary)
                time_rows.extend(rows)

    return summaries, time_rows


def run_all(seed_count: int = 3, quick: bool = True) -> None:
    ensure_dirs()

    seeds = list(range(seed_count))
    all_summaries: List[Dict] = []
    all_time_rows: List[Dict] = []

    for runner in [
        run_e1_partition_progress,
        run_e2_loss_vs_latency,
        run_e3_healing_after_reconnect,
    ]:
        summaries, rows = runner(seeds)
        all_summaries.extend(summaries)
        all_time_rows.extend(rows)

    summaries, rows = run_e4_scalability_overhead(seeds, quick=quick)
    all_summaries.extend(summaries)
    all_time_rows.extend(rows)

    summary_df = pd.DataFrame(all_summaries)
    time_df = pd.DataFrame(all_time_rows)

    summary_path = Path("results/raw/summary.csv")
    time_path = Path("results/raw/time_series.csv")

    summary_df.to_csv(summary_path, index=False)
    time_df.to_csv(time_path, index=False)

    print(f"Wrote {summary_path} with {len(summary_df)} rows")
    print(f"Wrote {time_path} with {len(time_df)} rows")

    print("\nSummary by experiment/protocol:")

    """
    cols = [
        "actuation_availability",
        "commit_availability",
        "mean_L_act_ms",
        "mean_L_commit_ms",
        "L_conv_ms",
        "healing_ms_after_partition",
        "delta_decision_final",
        "messages_sent",
        "bytes_sent",
        "active_state_items_mean",
    ]
    """

    cols = [
        "actuation_availability",
        "commit_availability",
        "mean_L_act_ms",
        "mean_L_commit_ms",
        "L_conv_ms",
        "healing_ms_after_partition",
        "convergence_fraction_final",
        "decided_fraction_final",
        "unresolved_fraction_final",
        "delta_decision_final",
        "messages_sent",
        "bytes_sent",
        "active_state_items_mean",
    ]

    grouped = (
        summary_df
        .groupby(["experiment", "protocol"])[cols]
        .median(numeric_only=True)
        .reset_index()
    )

    print(grouped.to_string(index=False))