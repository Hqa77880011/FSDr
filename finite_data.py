"""Finite learner/practice process; private skill never becomes an observed feature."""

import argparse
import csv
from pathlib import Path

import numpy as np


def learner(seed, learner_id, steps=80):
    rng = np.random.default_rng(np.random.SeedSequence([seed, learner_id]))
    skill = rng.normal(0.5, 0.5, 3)
    decay = rng.uniform(0.003, 0.02, 3)
    successes, attempts = np.ones(3), np.ones(3) * 2
    clock = 0.0
    rows = []
    difficulty = np.array([-0.4, 0.4, -0.2, 0.6, -0.5, 0.3])
    for _ in range(steps):
        gap = rng.exponential(3) + (36 if rng.random() < 0.12 else 0)
        clock += gap
        skill *= np.exp(-decay * gap)
        # Behavior depends only on available past outcomes, never private skill.
        need = 1 - successes / attempts
        probability = np.repeat(need, 2) + 0.5
        probability /= probability.sum()
        action = int(rng.choice(6, p=probability))
        concept = action // 2
        success_probability = 1 / (1 + np.exp(-(skill[concept] - difficulty[action])))
        response = int(rng.random() < success_probability)
        rows.append(dict(learner=f"learner-{learner_id:03d}", exercise=f"exercise-{action}",
                         concept=f"concept-{concept}", correct=response, timestamp=clock * 3600))
        attempts[concept] += 1
        successes[concept] += response
        skill[concept] += 0.12 * (1 - np.tanh(skill[concept]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learners", type=int, default=60)
    args = parser.parse_args()
    if args.learners < 20:
        parser.error("At least 20 independent learners required for the registered folds")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["learner", "exercise", "concept", "correct", "timestamp"])
        writer.writeheader()
        for identity in range(args.learners):
            writer.writerows(learner(args.seed, identity))


if __name__ == "__main__":
    main()
