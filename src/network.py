from __future__ import annotations

import math
import random
from typing import List


class Network:
    """
    Shared network model for all protocols.

    v0.1 model:
    - fully connected outside partition window
    - deterministic bisection during partition window
    - stochastic loss and log-normal delivery delay
    """

    def __init__(
        self,
        n: int,
        rng: random.Random,
        packet_loss: float = 0.0,
        delay_median_ms: float = 30.0,
        delay_sigma: float = 0.35,
        gossip_fanout: int = 2,
        partition_start_ms: int | None = None,
        partition_end_ms: int | None = None,
    ) -> None:
        self.n = n
        self.rng = rng
        self.packet_loss = packet_loss
        self.delay_median_ms = delay_median_ms
        self.delay_sigma = delay_sigma
        self.gossip_fanout = gossip_fanout
        self.partition_start_ms = partition_start_ms
        self.partition_end_ms = partition_end_ms

    def in_partition(self, time_ms: int) -> bool:
        if self.partition_start_ms is None or self.partition_end_ms is None:
            return False
        return self.partition_start_ms <= time_ms < self.partition_end_ms

    def component_id(self, robot: int) -> int:
        """
        Deterministic bisection:
        first half in component 0, second half in component 1.
        For even N this can remove majority connectivity.
        """
        return 0 if robot < self.n // 2 else 1

    def connected(self, src: int, dst: int, time_ms: int) -> bool:
        if src == dst:
            return True
        if not self.in_partition(time_ms):
            return True
        return self.component_id(src) == self.component_id(dst)

    def component_size(self, robot: int, time_ms: int) -> int:
        if not self.in_partition(time_ms):
            return self.n

        cid = self.component_id(robot)
        return sum(1 for r in range(self.n) if self.component_id(r) == cid)

    def has_majority(self, robot: int, time_ms: int) -> bool:
        return self.component_size(robot, time_ms) >= (self.n // 2 + 1)

    def should_drop(self) -> bool:
        return self.rng.random() < self.packet_loss

    def sample_delay_ms(self) -> int:
        """
        Log-normal delay with configured median.
        """
        mu = math.log(self.delay_median_ms)
        delay = self.rng.lognormvariate(mu, self.delay_sigma)
        return max(1, int(round(delay)))

    def peers(self, robot: int, time_ms: int) -> List[int]:
        return [
            r for r in range(self.n)
            if r != robot and self.connected(robot, r, time_ms)
        ]

    def gossip_peers(self, robot: int, time_ms: int) -> List[int]:
        candidates = self.peers(robot, time_ms)
        if len(candidates) <= self.gossip_fanout:
            return candidates
        return self.rng.sample(candidates, self.gossip_fanout)