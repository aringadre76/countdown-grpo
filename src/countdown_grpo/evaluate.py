"""Zero-shot baseline evaluation entry point."""

from __future__ import annotations

import argparse

from .data import load_countdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, test = load_countdown(seed=args.seed)
    print(f"Prepared {min(args.limit, len(test))} held-out tasks.")
    print("Model generation is intentionally not hidden behind this scaffold yet.")
    print("Next step: add deterministic Transformers generation and report verifier metrics.")


if __name__ == "__main__":
    main()
