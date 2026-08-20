from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "train.py"
    for seed in args.seeds:
        for subject in args.subjects:
            command = [
                sys.executable,
                str(train_script),
                "--config",
                args.config,
                "--subject",
                subject,
                "--seed",
                str(seed),
            ]
            subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
