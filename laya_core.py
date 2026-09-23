#!/usr/bin/env python3
"""Shared core: batched Laya scoring and metrics.

The scoring protocol is NOT invented here. `score_cases`, `softmax_t`, `temp_for`, `macro_f1` and
`metrics` are a faithful transcription of `research/scripts/bench_local.py` from the Laya repo, so
numbers here are comparable to the published ones. Any deviation is commented on its line.

Portions adapted from convaiinnovations/laya (research/scripts/bench_local.py), Apache-2.0.
Changes: the CPU/GPU synchronize guard in score_cases, the confidence_curve helper, and load().
"""
import json, math, os, sys, time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import laya
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket


def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


@torch.no_grad()
def score_cases(agent, cases, max_tokens=8192, max_seqs=64, tag=""):
    """Returns (logits_per_question, index, seconds, dropped). Copied from the official bench."""
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, mk = build_sequence(agent.tok, state, q, max_len, hml)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            if len(mk) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            items.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(mk)))
    order = sorted([i for i, it in enumerate(items) if it is not None],
                   key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, i = time.time(), 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and \
                max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"])); j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        lg, _ = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                            b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                            b["qtype"].to(agent.device))
        lg = lg.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = lg[r, :len(sel[r]["markers"])]
        i = j
    # Only synchronize when there IS a GPU: on a CPU-only torch, cuda.synchronize() raises
    # "Torch not compiled with CUDA enabled" after the whole run.
    if torch.cuda.is_available() and str(agent.device) != "cpu":
        torch.cuda.synchronize()
    return out, index, time.time() - t0, dropped


def softmax_t(z, t=1.0):
    z = np.asarray(z, float) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_for(agent, qt, k):
    """Temperature calibration per (question type, number of options), as shipped in the checkpoint."""
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


def macro_f1(g, p):
    g, p = np.asarray(g), np.asarray(p)
    f = []
    for c in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def ece_score(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf):
        return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def metrics(rows):
    """rows = [(gold_idx, prob_vector | None)]"""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0}
    g = np.array([x[0] for x in rows]); p = np.array([int(np.argmax(x[1])) for x in rows])
    c = np.array([float(np.max(x[1])) for x in rows]); corr = (p == g).astype(float)
    return {"n": len(rows), "accuracy": round(float(corr.mean()), 4),
            "macro_f1": round(macro_f1(g, p), 4), "ece": round(ece_score(c, corr), 4),
            "mean_confidence": round(float(c.mean()), 4),
            "acc_at_50_coverage": round(float(corr[np.argsort(-c)[:max(1, len(c)//2)]].mean()), 4)}


def confidence_curve(conf, corr, bands=((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.95),
                                        (0.95, 0.99), (0.99, 1.01))):
    """Accuracy per confidence band, with n of M in each band."""
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    M = len(conf)
    out = []
    for lo, hi in bands:
        s = (conf >= lo) & (conf < hi)
        n = int(s.sum())
        out.append({"band": "[%.2f, %.2f)" % (lo, hi), "n": n, "M": M,
                    "accuracy": round(float(corr[s].mean()), 4) if n else None,
                    "coverage": round(n / M, 4) if M else None})
    return out


def load(variant, device):
    """variant: 'english' | 'multilingual' | 'typed-decisions'"""
    sub = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}[variant]
    ag = laya.load("convaiinnovations/laya", subfolder=sub, device=device) if sub \
        else laya.load("convaiinnovations/laya", device=device)
    ag.model.eval()
    return ag
