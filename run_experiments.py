import argparse

from src.experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run larger E4 N sweep. Quick mode is used by default.",
    )

    args = parser.parse_args()
    run_all(seed_count=args.seeds, quick=not args.full)


if __name__ == "__main__":
    main()