from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

from .model import Action, Epoch, Message, Proposal
from .network import Network
from .protocols import ProtocolBase


@dataclass(order=True)
class Event:
    time_ms: int
    seq: int
    callback: Callable = field(compare=False)


class Simulator:
    def __init__(
        self,
        n: int,
        protocol_cls: Type[ProtocolBase],
        duration_ms: int,
        seed: int,
        packet_loss: float = 0.0,
        gossip_period_ms: int = 50,
        gossip_fanout: int = 2,
        partition_start_ms: Optional[int] = None,
        partition_end_ms: Optional[int] = None,
    ) -> None:
        self.n = n
        self.protocol_cls = protocol_cls
        self.duration_ms = duration_ms
        self.seed = seed
        self.rng = random.Random(seed)
        self.now_ms = 0
        self.gossip_period_ms = gossip_period_ms

        # Shared Raft abstraction state.
        self.raft_leader = 0
        self.raft_committed: Dict[Epoch, Action] = {}
        self.raft_commit_index = 0

        self.network = Network(
            n=n,
            rng=self.rng,
            packet_loss=packet_loss,
            gossip_fanout=gossip_fanout,
            partition_start_ms=partition_start_ms,
            partition_end_ms=partition_end_ms,
        )

        self.robots: List[ProtocolBase] = [
            protocol_cls(robot_id=i, sim=self) for i in range(n)
        ]

        self._queue: List[Event] = []
        self._seq = 0

        self.proposals_total = 0
        self.blocked_total = 0
        self.actuation_latencies: List[int] = []
        self.commit_latencies: List[int] = []
        self.messages_sent = 0
        self.bytes_sent = 0
        self.first_proposal_time_by_epoch: Dict[Epoch, int] = {}

    def schedule(self, delay_ms: int, callback: Callable) -> None:
        self.schedule_at(self.now_ms + max(0, delay_ms), callback)

    def schedule_at(self, time_ms: int, callback: Callable) -> None:
        self._seq += 1
        heapq.heappush(self._queue, Event(time_ms, self._seq, callback))

    def send(
        self,
        src: int,
        dst: int,
        kind: str,
        payload,
        size_bytes: int,
        protocol_name: str,
    ) -> None:
        if not self.network.connected(src, dst, self.now_ms):
            return

        if self.network.should_drop():
            return

        delay = self.network.sample_delay_ms()

        msg = Message(
            src=src,
            dst=dst,
            kind=kind,
            payload=payload,
            size_bytes=size_bytes,
        )

        self.record_message(protocol_name, size_bytes)

        def deliver() -> None:
            # Check connectivity again at delivery time.
            if self.network.connected(src, dst, self.now_ms):
                self.robots[dst].on_message(msg)

        self.schedule(delay, deliver)

    def record_message(self, protocol_name: str, size_bytes: int) -> None:
        self.messages_sent += 1
        self.bytes_sent += size_bytes

    def record_blocked(
        self,
        protocol_name: str,
        robot: int,
        epoch: Epoch,
        propose_time: int,
    ) -> None:
        self.blocked_total += 1

    def record_commit(
        self,
        protocol_name: str,
        robot: int,
        epoch: Epoch,
        propose_time: int,
        commit_time: int,
    ) -> None:
        self.commit_latencies.append(commit_time - propose_time)

    def record_actuation(
        self,
        protocol_name: str,
        robot: int,
        epoch: Epoch,
        propose_time: int,
        actuation_time: int,
    ) -> None:
        self.actuation_latencies.append(actuation_time - propose_time)

    def schedule_proposal(self, proposal: Proposal) -> None:
        def do_propose() -> None:
            self.proposals_total += 1

            existing = self.first_proposal_time_by_epoch.get(proposal.epoch)
            if existing is None or proposal.time_ms < existing:
                self.first_proposal_time_by_epoch[proposal.epoch] = proposal.time_ms

            self.robots[proposal.robot].propose(proposal.epoch, proposal.action)

        self.schedule_at(proposal.time_ms, do_propose)

    def start(self) -> None:
        for robot in self.robots:
            robot.start()

    def run(self) -> None:
        self.start()

        while self._queue:
            event = heapq.heappop(self._queue)

            if event.time_ms > self.duration_ms:
                break

            self.now_ms = event.time_ms
            event.callback()

    def decisions(self, epoch: Epoch) -> Dict[int, Optional[Action]]:
        return {
            i: robot.decide(epoch)
            for i, robot in enumerate(self.robots)
        }

    def action_counts(self, epoch: Epoch) -> Dict[Action, int]:
        counts: Dict[Action, int] = {}

        for decision in self.decisions(epoch).values():
            if decision is None:
                continue
            counts[decision] = counts.get(decision, 0) + 1

        return counts

    def decided_fraction(self, epoch: Epoch) -> float:
        decided = sum(
            1
            for decision in self.decisions(epoch).values()
            if decision is not None
        )
        return decided / self.n

    def unresolved_fraction(self, epoch: Epoch) -> float:
        return 1.0 - self.decided_fraction(epoch)

    def convergence_fraction(self, epoch: Epoch) -> float:
        """
        Fraction of the whole swarm that currently agrees on the dominant
        non-null decision.

        This is used for 95% convergence. It correctly treats no-decision
        states as unresolved rather than as agreement.
        """
        counts = self.action_counts(epoch)

        if not counts:
            return 0.0

        return max(counts.values()) / self.n

    def decision_agreement_among_decided(self, epoch: Epoch) -> float:
        """
        Agreement fraction among robots that have a non-null decision.

        Returns NaN if nobody has decided.
        """
        counts = self.action_counts(epoch)
        decided = sum(counts.values())

        if decided == 0:
            return float("nan")

        return max(counts.values()) / decided

    def decision_divergence(self, epoch: Epoch) -> float:
        """
        Disagreement among robots that have a non-null decision.

        This intentionally does not treat Raft unavailability as divergence.
        If nobody has decided, divergence is undefined rather than 1.0.
        """
        agreement = self.decision_agreement_among_decided(epoch)

        # NaN check.
        if agreement != agreement:
            return float("nan")

        return 1.0 - agreement

    def active_state_items_mean(self) -> float:
        return sum(
            robot.active_state_items()
            for robot in self.robots
        ) / self.n

    def sample_row(
        self,
        experiment: str,
        epoch: Epoch,
    ) -> Dict[str, float | int | str]:
        return {
            "experiment": experiment,
            "protocol": self.protocol_cls.name,
            "seed": self.seed,
            "n": self.n,
            "time_ms": self.now_ms,
            "epoch": epoch,
            "convergence_fraction": self.convergence_fraction(epoch),
            "decided_fraction": self.decided_fraction(epoch),
            "unresolved_fraction": self.unresolved_fraction(epoch),
            "agreement_among_decided": self.decision_agreement_among_decided(epoch),
            "delta_decision": self.decision_divergence(epoch),
            "messages_sent": self.messages_sent,
            "bytes_sent": self.bytes_sent,
            "active_state_items_mean": self.active_state_items_mean(),
        }

    def summary(self, epoch: Epoch = 1) -> Dict[str, float | int | str]:
        protocol = self.protocol_cls.name

        actuation_availability = (
            len(self.actuation_latencies) / self.proposals_total
            if self.proposals_total
            else 0.0
        )

        commit_availability = (
            len(self.commit_latencies) / self.proposals_total
            if self.proposals_total and protocol == "Raft"
            else float("nan")
        )

        mean_lact = (
            sum(self.actuation_latencies) / len(self.actuation_latencies)
            if self.actuation_latencies
            else float("nan")
        )

        mean_lcommit = (
            sum(self.commit_latencies) / len(self.commit_latencies)
            if self.commit_latencies
            else float("nan")
        )

        return {
            "protocol": protocol,
            "seed": self.seed,
            "n": self.n,
            "proposals": self.proposals_total,
            "blocked": self.blocked_total,
            "commit_availability": commit_availability,
            "actuation_availability": actuation_availability,
            "mean_L_act_ms": mean_lact,
            "mean_L_commit_ms": mean_lcommit,
            "convergence_fraction_final": self.convergence_fraction(epoch),
            "decided_fraction_final": self.decided_fraction(epoch),
            "unresolved_fraction_final": self.unresolved_fraction(epoch),
            "agreement_among_decided_final": self.decision_agreement_among_decided(epoch),
            "delta_decision_final": self.decision_divergence(epoch),
            "messages_sent": self.messages_sent,
            "bytes_sent": self.bytes_sent,
            "active_state_items_mean": self.active_state_items_mean(),
        }