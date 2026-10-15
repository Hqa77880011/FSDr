import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .method import ForgettingOutcome, estimates, policy


def prepare(args):
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    exercises = sorted({r["exercise"] for r in rows})
    concepts = sorted({r["concept"] for r in rows})
    mapping, grouped = {}, {}
    exercise_index = {a: i for i, a in enumerate(exercises)}
    concept_index = {c: i for i, c in enumerate(concepts)}
    for row in rows:
        a, c = row["exercise"], concept_index[row["concept"]]
        if a in mapping and mapping[a] != c:
            raise ValueError("Each exercise needs one fixed concept mapping")
        mapping[a] = c
        grouped.setdefault(row["learner"], []).append(row)
    learners = sorted(grouped)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(learners)
    ntrain, ncal = int(0.7 * len(learners)), int(0.1 * len(learners))
    split = {u: "train" if i < ntrain else "cal" if i < ntrain + ncal else "test" for i, u in enumerate(shuffled)}
    packed = {}
    for learner in learners:
        events = sorted(grouped[learner], key=lambda r: float(r["timestamp"]))
        actions = np.array([exercise_index[r["exercise"]] for r in events])
        rewards = np.array([float(r["correct"]) for r in events])
        times = np.array([float(r["timestamp"]) for r in events]) / args.time_unit_seconds
        if not np.isin(rewards, [0, 1]).all():
            raise ValueError("Correctness outcomes must be binary")
        for begin in range(0, len(events) - args.horizon + 1, args.horizon):
            history, response, ages = [], [], []
            for t in range(begin, begin + args.horizon + 1):
                current_time = times[t] if t < len(times) else times[-1] + 1 / args.time_unit_seconds
                old = np.arange(max(0, t - args.history), t)
                h = np.full(args.history, -1, int)
                r, age = np.zeros(args.history), np.zeros(args.history)
                if len(old):
                    h[-len(old):], r[-len(old):], age[-len(old):] = actions[old], rewards[old], current_time - times[old]
                history.append(h)
                response.append(r)
                ages.append(age)
            packed.setdefault(split[learner], []).append((history, response, ages, actions[begin:begin + args.horizon], rewards[begin:begin + args.horizon], learner))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, samples in packed.items():
        history, response, age, action, reward, learner = zip(*samples)
        np.savez_compressed(out / f"{name}.npz", history=np.array(history), response=np.array(response, np.float32),
                            age=np.array(age, np.float32), action=action, reward=np.array(reward, np.float32), learner=learner)
    (out / "actions.json").write_text(json.dumps({"exercises": exercises, "concepts": [mapping[a] for a in exercises], "horizon": args.horizon}, indent=2))


def state_tensors(data, episode, step, device):
    return (torch.as_tensor(data["history"][episode, step], dtype=torch.long, device=device),
            torch.as_tensor(data["response"][episode, step], device=device),
            torch.as_tensor(data["age"][episode, step], device=device),
            torch.as_tensor(step, dtype=torch.long, device=device))


def target_policy(data, episode, step, concepts, kind, popularity):
    return policy(data["history"][episode, step], data["response"][episode, step], concepts, kind, popularity)


def fit(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    metadata = json.loads((root / "actions.json").read_text())
    data, calibration = [np.load(root / f"{split}.npz", allow_pickle=False) for split in ["train", "cal"]]
    concepts, horizon = metadata["concepts"], metadata["horizon"]
    counts = np.bincount(data["action"].ravel(), minlength=len(concepts)) + 1
    popularity = counts / counts.sum()
    learners = rng.permutation(np.unique(data["learner"]))
    if len(learners) < args.folds:
        raise ValueError("Not enough independent learners for the requested folds")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for kind in args.policies:
        target_dir = out / kind
        target_dir.mkdir(exist_ok=True)
        for fold, heldout in enumerate(np.array_split(learners, args.folds)):
            episodes = np.flatnonzero(~np.isin(data["learner"], heldout))
            net = ForgettingOutcome(concepts, args.width, args.mode, horizon).to(args.device)
            target_net = copy.deepcopy(net).eval()
            optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
            for _ in range(args.epochs):
                states = rng.permutation(len(episodes) * horizon)
                for start in range(0, len(states), args.batch_size):
                    selected = states[start:start + args.batch_size]
                    ep, step = episodes[selected // horizon], selected % horizon
                    q, logits, reward_logits = net(*state_tensors(data, ep, step, args.device))
                    action = torch.as_tensor(data["action"][ep, step], device=args.device)
                    reward = torch.as_tensor(data["reward"][ep, step], device=args.device)
                    pi_next = torch.as_tensor(target_policy(data, ep, step + 1, concepts, kind, popularity), dtype=torch.float32, device=args.device)
                    with torch.no_grad():
                        next_q = target_net(*state_tensors(data, ep, step + 1, args.device))[0]
                        next_v = (next_q * pi_next).sum(1) * torch.as_tensor(step < horizon - 1, device=args.device)
                        target = reward + 0.95 * next_v
                    row = torch.arange(len(ep), device=args.device)
                    loss = F.mse_loss(q[row, action], target) + F.cross_entropy(logits, action)
                    loss = loss + 0.2 * F.binary_cross_entropy_with_logits(reward_logits[row, action], reward)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for slow, fast in zip(target_net.parameters(), net.parameters()):
                            slow.lerp_(fast, 0.01)
            net.eval()
            calibration_logits, calibration_actions = [], []
            with torch.no_grad():
                for ep in range(len(calibration["action"])):
                    episodes_array = np.full(horizon, ep)
                    states = np.arange(horizon)
                    calibration_logits.append(net(*state_tensors(calibration, episodes_array, states, args.device))[1].cpu().numpy())
                    calibration_actions.extend(calibration["action"][ep])
            cl = np.concatenate(calibration_logits)
            ca = np.array(calibration_actions)
            result = minimize_scalar(lambda log_t: float(-np.log(np.maximum(softmax(cl / np.exp(log_t), axis=1)[np.arange(len(ca)), ca], 1e-12)).mean()), bounds=(-3, 3), method="bounded")
            torch.save({"model": net.state_dict(), "concepts": concepts, "horizon": horizon, "width": args.width,
                        "mode": args.mode, "temperature": float(np.exp(result.x)), "popularity": torch.tensor(popularity),
                        "training_learners": np.unique(data["learner"][episodes]).tolist()}, target_dir / f"fold_{fold}.pt")


def evaluate(args):
    rng = np.random.default_rng(args.seed)
    data = np.load(Path(args.data) / "test.npz", allow_pickle=False)
    learners = np.unique(data["learner"])
    assignment = {u: i % args.folds for i, u in enumerate(rng.permutation(learners))}
    bootstrap = [np.concatenate([np.flatnonzero(data["learner"] == u) for u in rng.choice(learners, len(learners), replace=True)]) for _ in range(args.bootstrap)]
    results, contributions = {}, {}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads(Path(args.truth).read_text()) if args.truth else None
    for kind in args.policies:
        mu, pi, q_all, v_next, v_initial, reward_probability, gaps = [], [], [], [], [], [], []
        fitted = {}
        for fold in range(args.folds):
            saved = torch.load(Path(args.models) / kind / f"fold_{fold}.pt", map_location=args.device, weights_only=True)
            net = ForgettingOutcome(saved["concepts"], saved["width"], saved["mode"], saved["horizon"]).to(args.device)
            net.load_state_dict(saved["model"])
            fitted[fold] = saved, net.eval()
        for i in range(len(data["action"])):
            fold = assignment[str(data["learner"][i])]
            saved, net = fitted[fold]
            if str(data["learner"][i]) in saved["training_learners"]:
                raise ValueError("A scored learner was used for nuisance fitting")
            horizon = saved["horizon"]
            ep, steps = np.full(horizon + 1, i), np.arange(horizon + 1)
            with torch.no_grad():
                q, logits, reward_logits = net(*state_tensors(data, ep, steps, args.device))
            target = target_policy(data, ep, steps, saved["concepts"], kind, saved["popularity"].cpu().numpy())
            q = q.cpu().numpy()
            value = (q * target).sum(1)
            value[-1] = 0
            propensity = softmax(logits.cpu().numpy() / saved["temperature"], axis=1)
            mu.append(propensity[:-1])
            pi.append(target[:-1])
            q_all.append(q[:-1])
            v_next.append(value[1:])
            v_initial.append(value[0])
            action = data["action"][i]
            reward_probability.extend(torch.sigmoid(reward_logits[:-1])[torch.arange(horizon, device=args.device), torch.as_tensor(action, device=args.device)].cpu().numpy())
            for t in range(horizon):
                same = (data["history"][i, t] >= 0)
                query_concept = saved["concepts"][action[t]]
                same &= np.array([saved["concepts"][a] if a >= 0 else -1 for a in data["history"][i, t]]) == query_concept
                gaps.append(float(data["age"][i, t][same].min()) if same.any() else float("inf"))
        mu, pi, q_all, v_next, v_initial = [np.asarray(v) for v in [mu, pi, q_all, v_next, v_initial]]
        samples, snips, clipped, ess, weights = estimates(data["reward"], data["action"], mu, pi, q_all, v_next, v_initial)
        min_mu = float(mu[pi > args.eta].min()) if np.any(pi > args.eta) else 0
        sensitivity = max(float(abs(a.mean() - b.mean())) for a in clipped.values() for b in clipped.values())
        release = min_mu >= args.min_propensity and float(ess.min()) >= args.min_ess and sensitivity <= args.max_clipping_change
        statistics = {}
        for name, values in samples.items():
            boot = np.array([values[index].mean() for index in bootstrap])
            statistics[name] = {"value": float(values.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
        snips_boot = []
        discount = 0.95 ** np.arange(data["reward"].shape[1])
        for index in bootstrap:
            snips_boot.append(float(np.sum(discount * (weights[index] * data["reward"][index]).sum(0) / np.maximum(weights[index].sum(0), 1e-300)) / discount.sum()))
        statistics["SNIPS"] = {"value": snips, "ci95": np.quantile(snips_boot, [0.025, 0.975]).tolist()}
        reward = data["reward"].ravel()
        probability = np.array(reward_probability)
        long = np.array(gaps) >= args.long_gap
        auc = float(roc_auc_score(reward, probability)) if len(np.unique(reward)) == 2 else None
        long_auc = float(roc_auc_score(reward[long], probability[long])) if len(np.unique(reward[long])) == 2 else None
        results[kind] = {"released": bool(release), "released_dr": statistics["DR"] if release else None,
                         "diagnostics": statistics, "minimum_propensity": min_mu, "minimum_ess_ratio": float(ess.min()),
                         "clipping_change": sensitivity, "reward_auc": auc, "long_gap_auc": long_auc}
        if truth is not None:
            for name, stat in statistics.items():
                stat["absolute_bias"] = abs(stat["value"] - truth[kind])
                stat["squared_error"] = (stat["value"] - truth[kind]) ** 2
                stat["covers_truth"] = bool(stat["ci95"][0] <= truth[kind] <= stat["ci95"][1])
        contributions[kind] = samples["DR"]
    comparisons = {}
    policies = list(contributions)
    for i, a in enumerate(policies):
        for b in policies[i + 1:]:
            difference = contributions[a] - contributions[b]
            interval = np.quantile([difference[index].mean() for index in bootstrap], [0.025, 0.975])
            supported = results[a]["released"] and results[b]["released"]
            comparisons[a + " vs " + b] = {"released": supported,
                                           "difference": float(difference.mean()) if supported else None,
                                           "ci95": interval.tolist() if supported else None,
                                           "diagnostic_difference": float(difference.mean())}
    rank_correlation = None
    if truth and len(policies) >= 2:
        correlation = spearmanr([truth[p] for p in policies], [contributions[p].mean() for p in policies]).statistic
        rank_correlation = float(correlation) if np.isfinite(correlation) else None
    (out / "report.json").write_text(json.dumps({"policies": results, "comparisons": comparisons, "diagnostic_rank_spearman": rank_correlation}, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(policies, [results[p]["minimum_ess_ratio"] for p in policies])
    ax.axhline(args.min_ess, color="gray", linestyle="--")
    ax.set(ylabel="Minimum horizon ESS ratio")
    fig.tight_layout()
    fig.savefig(out / "support.png", dpi=180)
    plt.close(fig)


def summarize(args):
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    result = {}
    for policy_name in reports[0]["policies"]:
        result[policy_name] = {}
        for estimator in reports[0]["policies"][policy_name]["diagnostics"]:
            values = [report["policies"][policy_name]["diagnostics"][estimator] for report in reports]
            if any("squared_error" not in value for value in values):
                raise ValueError("Bias/RMSE aggregation requires independent reports with supplied oracle truth")
            result[policy_name][estimator] = {"mean_absolute_bias": float(np.mean([v["absolute_bias"] for v in values])),
                                              "rmse": float(np.sqrt(np.mean([v["squared_error"] for v in values]))),
                                              "ci95_coverage": float(np.mean([v["covers_truth"] for v in values]))}
    correlations = [r["diagnostic_rank_spearman"] for r in reports if r["diagnostic_rank_spearman"] is not None]
    result["mean_rank_spearman"] = float(np.mean(correlations)) if correlations else None
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--time-unit-seconds", type=float, default=3600)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(function=prepare)
    p = sub.add_parser("summarize")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", default="summary.json")
    p.set_defaults(function=summarize)
    for command in ["fit", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--policies", nargs="+", choices=["uniform", "popularity", "gap", "aggressive"], default=["uniform", "popularity", "gap", "aggressive"])
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cpu")
        if command == "fit":
            p.add_argument("--mode", choices=["personalized", "time", "global", "no_time"], default="personalized")
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--width", type=int, default=48)
            p.add_argument("--batch-size", type=int, default=128)
            p.add_argument("--lr", type=float, default=1e-3)
            p.set_defaults(function=fit)
        else:
            p.add_argument("--models", default="runs/fit")
            p.add_argument("--bootstrap", type=int, default=1000)
            p.add_argument("--eta", type=float, default=0.01)
            p.add_argument("--min-propensity", type=float, default=0.02)
            p.add_argument("--min-ess", type=float, default=0.1)
            p.add_argument("--max-clipping-change", type=float, default=0.02)
            p.add_argument("--long-gap", type=float, default=24)
            p.add_argument("--truth")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
