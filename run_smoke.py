from src.model import Proposal
from src.protocols import CRDTIntent, GossipOverwrite, RaftIntent
from src.simulator import Simulator


def build_workload(n: int) -> list[Proposal]:
    """
    Smoke workload:
    - proposals before partition
    - proposals during partition
    - proposals after partition
    """
    proposals: list[Proposal] = []

    actions = ["a1", "a2"]

    for i in range(n):
        proposals.append(Proposal(time_ms=1000 + i * 10, robot=i, epoch=1, action=actions[i % 2]))

    for i in range(n):
        proposals.append(Proposal(time_ms=6000 + i * 10, robot=i, epoch=2, action=actions[(i + 1) % 2]))

    for i in range(n):
        proposals.append(Proposal(time_ms=12000 + i * 10, robot=i, epoch=3, action=actions[i % 2]))

    return proposals


def run_protocol(protocol_cls, seed: int = 7) -> None:
    n = 6
    sim = Simulator(
        n=n,
        protocol_cls=protocol_cls,
        duration_ms=20000,
        seed=seed,
        packet_loss=0.05,
        gossip_period_ms=50,
        gossip_fanout=2,
        partition_start_ms=5000,
        partition_end_ms=10000,
    )

    for proposal in build_workload(n):
        sim.schedule_proposal(proposal)

    sim.run()

    print("\n===", protocol_cls.name, "===")
    print(sim.summary(epoch=3))
    print("Final decisions epoch 3:", sim.decisions(epoch=3))


if __name__ == "__main__":
    for cls in [RaftIntent, GossipOverwrite, CRDTIntent]:
        run_protocol(cls)