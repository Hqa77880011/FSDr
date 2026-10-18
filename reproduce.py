"""Chronological learner logs; horizon/history20, five learner folds, width48, 30 epochs seed42, support-gated normalized DR and learner bootstrap1000."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "fsdr", "prepare", "--csv", args.csv, "--output", str(output / "data"), "--horizon", "20", "--history", "20", "--time-unit-seconds", "3600", "--seed", "42"],
        ["-m", "fsdr", "fit", "--data", str(output / "data"), "--output", str(output / "models"), "--folds", "5", "--epochs", "30", "--width", "48", "--seed", "42"],
        ["-m", "fsdr", "evaluate", "--data", str(output / "data"), "--models", str(output / "models"), "--output", str(output / "evaluation"), "--folds", "5", "--bootstrap", "1000", "--seed", "42"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--csv', required=True, type=absolute_path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
