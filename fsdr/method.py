import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ForgettingOutcome(nn.Module):
    def __init__(self, concepts, width=48, mode="personalized", horizon=20):
        super().__init__()
        self.register_buffer("concepts", torch.tensor(concepts, dtype=torch.long))
        self.mode, self.horizon = mode, horizon
        actions, classes = len(concepts), max(concepts) + 1
        self.exercise = nn.Embedding(actions + 1, width, padding_idx=actions)
        self.response = nn.Embedding(2, width)
        self.behavior = nn.Linear(width + 1, actions)
        self.global_decay = nn.Parameter(torch.zeros(classes))
        self.learner_decay = nn.Linear(width, classes)
        self.q = nn.Sequential(nn.Linear(3 * width + 1, width), nn.ReLU(), nn.Linear(width, 1))
        self.reward = nn.Linear(3 * width + 1, 1)

    def forward(self, history, response, age, step):
        valid = history >= 0
        safe = history.clamp_min(0)
        tokens = self.exercise(torch.where(valid, safe, len(self.concepts))) + self.response(response.long())
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        queries = self.exercise.weight[:-1]
        logits = torch.einsum("ad,btd->bat", queries, tokens) / np.sqrt(tokens.shape[-1])
        if self.mode != "no_time":
            rates = self.global_decay[self.concepts][None].expand(len(history), -1)
            if self.mode == "personalized":
                shrinkage = valid.sum(1, keepdim=True) / (valid.sum(1, keepdim=True) + 10)
                rates = rates + shrinkage * self.learner_decay(pooled)[:, self.concepts]
            if self.mode == "global":
                rates = self.global_decay.mean().expand_as(rates)
            delay = age if self.mode == "global" else torch.log1p(age)
            logits = logits - F.softplus(rates)[..., None] * delay[:, None]
        attention = logits.masked_fill(~valid[:, None], -1e9).softmax(-1) * valid[:, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = attention @ tokens
        query = queries[None].expand(len(history), -1, -1)
        current = pooled[:, None].expand_as(query)
        remaining = (self.horizon - step.float()) / self.horizon
        inputs = torch.cat([query, context, current, remaining[:, None, None].expand(-1, len(queries), 1)], -1)
        maximum = (1 - 0.95 ** (self.horizon - step.float())) / 0.05
        value = self.q(inputs).squeeze(-1).sigmoid() * maximum[:, None]
        return value, self.behavior(torch.cat([pooled, remaining[:, None]], -1)), self.reward(inputs).squeeze(-1)


def policy(history, response, concepts, kind, popularity):
    actions = len(concepts)
    if kind == "uniform":
        return np.full((len(history), actions), 1 / actions)
    if kind == "popularity":
        return np.broadcast_to(popularity, (len(history), actions)).copy()
    score = np.ones((len(history), actions))
    for i, (past, outcome) in enumerate(zip(history, response)):
        valid = past >= 0
        past_concepts = np.asarray(concepts)[past[valid]]
        for a, concept in enumerate(concepts):
            mask = past_concepts == concept
            score[i, a] = (1 + np.sum(1 - outcome[valid][mask])) / (2 + mask.sum())
    if kind == "aggressive":
        result = np.full_like(score, 0.01 / actions)
        result[np.arange(len(score)), score.argmax(1)] += 0.99
        return result
    return score / score.sum(1, keepdims=True)


def estimates(rewards, action, propensity, target, q, next_value, initial_value, clips=(10, 20, 50), gamma=0.95):
    count, horizon = action.shape
    i, t = np.indices(action.shape)
    logged_mu, logged_pi = propensity[i, t, action], target[i, t, action]
    log_ratio = np.where(logged_pi > 0, np.log(np.maximum(logged_pi, 1e-300)) - np.log(np.maximum(logged_mu, 1e-300)), -np.inf)
    cumulative = np.cumsum(log_ratio, axis=1)
    finite_max = np.max(cumulative, axis=0)
    scaled = np.exp(cumulative - np.where(np.isfinite(finite_max), finite_max, 0))
    ess = scaled.sum(0) ** 2 / np.maximum((scaled ** 2).sum(0), 1e-300)
    discount = gamma ** np.arange(horizon)
    normalizer = discount.sum()
    dr = {}
    plugin = initial_value / normalizer
    chosen_q = q[i, t, action]
    residual = rewards + gamma * next_value - chosen_q
    for clip in clips:
        weights = np.exp(np.minimum(cumulative, np.log(clip)))
        dr[clip] = (initial_value + (weights * residual * discount).sum(1)) / normalizer
    weights = np.exp(np.minimum(cumulative, np.log(20)))
    ips = (weights * rewards * discount).sum(1) / normalizer
    snips = float(np.sum(discount * (weights * rewards).sum(0) / np.maximum(weights.sum(0), 1e-300)) / normalizer)
    return {"DR": dr[20], "DM_FQE": plugin, "IPS": ips}, snips, dr, ess / count, weights
