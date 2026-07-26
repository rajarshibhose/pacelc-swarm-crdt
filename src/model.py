from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RobotId = int
Epoch = int
Action = str


@dataclass(frozen=True, order=True)
class Vote:
    epoch: Epoch
    robot: RobotId
    counter: int
    action: Action


@dataclass(frozen=True)
class Message:
    src: RobotId
    dst: RobotId
    kind: str
    payload: Any
    size_bytes: int


@dataclass(frozen=True)
class Proposal:
    time_ms: int
    robot: RobotId
    epoch: Epoch
    action: Action