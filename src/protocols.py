from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict, Optional, Set

from .model import Action, Epoch, Message, RobotId, Vote

if TYPE_CHECKING:
    from .simulator import Simulator


class ProtocolBase:
    name = "base"

    def __init__(self, robot_id: RobotId, sim: Simulator) -> None:
        self.robot_id = robot_id
        self.sim = sim

    def start(self) -> None:
        pass

    def propose(self, epoch: Epoch, action: Action) -> None:
        raise NotImplementedError

    def on_message(self, msg: Message) -> None:
        raise NotImplementedError

    def decide(self, epoch: Epoch) -> Optional[Action]:
        raise NotImplementedError

    def active_state_items(self) -> int:
        return 0


class RaftIntent(ProtocolBase):
    """
    Commit-gated linearizable intent replication.

    This is a leader-and-quorum abstraction of Raft:
    - all proposals are routed to a single leader
    - a proposal commits only if the leader can reach a majority
    - if a proposal is blocked during a partition, it is retried after reconnect
    - local actuation happens only after commit
    - committed epoch decisions follow one shared global order
    """

    name = "Raft"

    def __init__(self, robot_id: RobotId, sim: Simulator) -> None:
        super().__init__(robot_id, sim)
        self.committed: Dict[Epoch, Action] = {}

    def propose(self, epoch: Epoch, action: Action) -> None:
        start = self.sim.now_ms
        self._attempt_commit(
            epoch=epoch,
            action=action,
            original_propose_time=start,
            count_blocked=True,
        )

    def _attempt_commit(
        self,
        epoch: Epoch,
        action: Action,
        original_propose_time: int,
        count_blocked: bool,
    ) -> None:
        leader = self.sim.raft_leader
        now = self.sim.now_ms

        client_can_reach_leader = (
            self.robot_id == leader
            or self.sim.network.connected(self.robot_id, leader, now)
        )

        leader_has_majority = self.sim.network.has_majority(leader, now)

        if not client_can_reach_leader or not leader_has_majority:
            if count_blocked:
                self.sim.record_blocked(
                    self.name,
                    self.robot_id,
                    epoch,
                    original_propose_time,
                )

            self._schedule_retry(
                epoch=epoch,
                action=action,
                original_propose_time=original_propose_time,
            )
            return

        peers = self.sim.network.peers(leader, now)
        majority = self.sim.n // 2 + 1
        quorum_followers_needed = max(0, majority - 1)
        quorum_followers = peers[:quorum_followers_needed]

        # Client request + response if proposer is not the leader.
        if self.robot_id != leader:
            self.sim.record_message(self.name, size_bytes=64)
            self.sim.record_message(self.name, size_bytes=64)

        # Abstract AppendEntries + acknowledgements to quorum followers.
        for _ in quorum_followers:
            self.sim.record_message(self.name, size_bytes=64)
            self.sim.record_message(self.name, size_bytes=64)

        commit_delay = 2 * self.sim.network.sample_delay_ms()

        def commit() -> None:
            self.sim.raft_commit_index += 1
            self.sim.raft_committed[epoch] = action

            # Expose committed value to robots reachable from the leader at
            # commit time. After reconnect, this becomes swarm-wide.
            visible_to = [leader] + self.sim.network.peers(leader, self.sim.now_ms)

            for robot_id in visible_to:
                peer = self.sim.robots[robot_id]
                if isinstance(peer, RaftIntent):
                    peer.committed[epoch] = action

            self.sim.record_commit(
                self.name,
                self.robot_id,
                epoch,
                original_propose_time,
                self.sim.now_ms,
            )
            self.sim.record_actuation(
                self.name,
                self.robot_id,
                epoch,
                original_propose_time,
                self.sim.now_ms,
            )

        self.sim.schedule(commit_delay, commit)

    def _schedule_retry(
        self,
        epoch: Epoch,
        action: Action,
        original_propose_time: int,
    ) -> None:
        partition_end = self.sim.network.partition_end_ms

        if partition_end is not None and self.sim.now_ms < partition_end:
            retry_time = partition_end + self.sim.network.sample_delay_ms()
        else:
            retry_time = self.sim.now_ms + 1000

        if retry_time > self.sim.duration_ms:
            return

        def retry() -> None:
            self._attempt_commit(
                epoch=epoch,
                action=action,
                original_propose_time=original_propose_time,
                count_blocked=False,
            )

        self.sim.schedule_at(retry_time, retry)

    def on_message(self, msg: Message) -> None:
        # Raft v0.1 uses an abstract leader/quorum commit model.
        pass

    def decide(self, epoch: Epoch) -> Optional[Action]:
        return self.committed.get(epoch)

    def active_state_items(self) -> int:
        return len(self.committed)
    

class GossipOverwrite(ProtocolBase):
    """
    Epidemic-gossip baseline:
    - same anti-entropy transport style as CRDT
    - no semilattice merge
    - stores one most-recent-by-arrival intent per epoch
    """

    name = "Gossip"

    def __init__(self, robot_id: RobotId, sim: Simulator) -> None:
        super().__init__(robot_id, sim)
        self.state: Dict[Epoch, Action] = {}

    def start(self) -> None:
        self.sim.schedule(self.sim.gossip_period_ms, self.gossip_tick)

    def propose(self, epoch: Epoch, action: Action) -> None:
        start = self.sim.now_ms
        self.state[epoch] = action
        self.sim.record_actuation(
            self.name,
            self.robot_id,
            epoch,
            start,
            self.sim.now_ms,
        )

    def gossip_tick(self) -> None:
        if self.sim.now_ms >= self.sim.duration_ms:
            return

        payload = dict(self.state)
        size = 16 + 16 * len(payload)

        for peer in self.sim.network.gossip_peers(self.robot_id, self.sim.now_ms):
            self.sim.send(
                src=self.robot_id,
                dst=peer,
                kind="gossip_state",
                payload=payload,
                size_bytes=size,
                protocol_name=self.name,
            )

        self.sim.schedule(self.sim.gossip_period_ms, self.gossip_tick)

    def on_message(self, msg: Message) -> None:
        if msg.kind != "gossip_state":
            return

        incoming: Dict[Epoch, Action] = msg.payload

        # Overwrite by arrival order.
        for epoch, action in incoming.items():
            self.state[epoch] = action

    def decide(self, epoch: Epoch) -> Optional[Action]:
        return self.state.get(epoch)

    def active_state_items(self) -> int:
        return len(self.state)


class CRDTIntent(ProtocolBase):
    """
    Intent-lattice CRDT:
    - vote set is grow-only
    - merge is set union
    - Decide uses latest vote per robot and deterministic majority resolver
    """

    name = "CRDT"

    def __init__(self, robot_id: RobotId, sim: Simulator) -> None:
        super().__init__(robot_id, sim)
        self.counter = 0
        self.votes: Set[Vote] = set()

    def start(self) -> None:
        self.sim.schedule(self.sim.gossip_period_ms, self.gossip_tick)

    def propose(self, epoch: Epoch, action: Action) -> None:
        start = self.sim.now_ms
        self.counter += 1
        self.votes.add(Vote(epoch, self.robot_id, self.counter, action))
        self.sim.record_actuation(
            self.name,
            self.robot_id,
            epoch,
            start,
            self.sim.now_ms,
        )

    def gossip_tick(self) -> None:
        if self.sim.now_ms >= self.sim.duration_ms:
            return

        payload = set(self.votes)
        size = 16 + 32 * len(payload)

        for peer in self.sim.network.gossip_peers(self.robot_id, self.sim.now_ms):
            self.sim.send(
                src=self.robot_id,
                dst=peer,
                kind="crdt_state",
                payload=payload,
                size_bytes=size,
                protocol_name=self.name,
            )

        self.sim.schedule(self.sim.gossip_period_ms, self.gossip_tick)

    def on_message(self, msg: Message) -> None:
        if msg.kind != "crdt_state":
            return

        incoming: Set[Vote] = msg.payload
        self.votes |= incoming

    def decide(self, epoch: Epoch) -> Optional[Action]:
        latest_by_robot: Dict[RobotId, Vote] = {}

        for vote in self.votes:
            if vote.epoch != epoch:
                continue

            current = latest_by_robot.get(vote.robot)

            if current is None:
                latest_by_robot[vote.robot] = vote
            elif vote.counter > current.counter:
                latest_by_robot[vote.robot] = vote
            elif vote.counter == current.counter and vote.action > current.action:
                # Deterministic tie-break.
                latest_by_robot[vote.robot] = vote

        if not latest_by_robot:
            return None

        counts = Counter(vote.action for vote in latest_by_robot.values())
        max_count = max(counts.values())
        winners = [action for action, count in counts.items() if count == max_count]

        # Deterministic resolver.
        return sorted(winners)[0]

    def active_state_items(self) -> int:
        return len(self.votes)