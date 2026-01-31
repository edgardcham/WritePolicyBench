from __future__ import annotations

import argparse

from .runner import Runner


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="writepolicybench",
        description="Streaming benchmark harness (scaffold).",
    )
    parser.add_argument("--config", default=None, help="Path to config file.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    args = parser.parse_args()

    runner = Runner()
    runner.run(episodes=args.episodes, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
