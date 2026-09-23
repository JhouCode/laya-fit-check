#!/usr/bin/env python3
"""Step 1: reproduce the published number before testing anything else.

Target: Laya's own benchmark artifact `research/results/t4_colab_benchmark.json`,
suites["massive_intent.en"]["laya"]["calibrated"]:
    accuracy 0.7833333 · macro_f1 0.7859905 · n 300
with meta {seed: 13, n_opts: 20, per_lang: 300}. The case builder below is a transcription of
`build_massive` from the official bench.

Inference is deterministic (one forward pass, argmax), so the only expected noise is kernel/device.
PASS if |accuracy - 0.783333| <= 0.02 (6 items of 300). Exit code 3 on FAIL: if this fails, don't
trust anything else you measure with this install.

    python reproduce_massive.py --device cuda      # or cpu
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_core import load, score_cases, softmax_t, temp_for, metrics

SEED, N_OPTS, PER_LANG = 13, 20, 300
TARGET_ACC, TARGET_F1, TOL = 0.7833333333333333, 0.7859904974658811, 0.02


def build_massive(per_lang):
    from datasets import load_dataset
    d = load_dataset("mteb/amazon_massive_intent", "en", split="test")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(SEED)
    cases, gold = [], []
    for r in list(d)[:per_lang]:
        pool = [x for x in labels if x != r["label_text"]]
        keys = [r["label_text"]] + rng.sample(pool, min(N_OPTS - 1, len(pool)))
        rng.shuffle(keys)
        cases.append(({"utterance": r["text"]},
                      {"intent": {"type": "choice",
                                  "instructions": "What is the user asking for in `utterance`?",
                                  "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}}}))
        gold.append(keys.index(r["label_text"]))
    return cases, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/massive_en.json")
    a = ap.parse_args()
    cases, gold = build_massive(PER_LANG)
    agent = load("english", a.device)
    logits, index, secs, dropped = score_cases(agent, cases)
    rows = [(gold[ci], softmax_t(z, temp_for(agent, qt, k)) if z is not None else None)
            for (ci, _, qt, k), z in zip(index, logits)]
    m = metrics(rows)
    m.update({"device": a.device, "seconds": round(secs, 2), "dropped": dropped,
              "target_accuracy": TARGET_ACC, "tolerance": TOL,
              "delta": round(m["accuracy"] - TARGET_ACC, 4),
              "verdict": "PASS" if abs(m["accuracy"] - TARGET_ACC) <= TOL else "FAIL",
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(m, open(a.out, "w"), indent=1)
    print(json.dumps(m, indent=1))
    sys.exit(0 if m["verdict"] == "PASS" else 3)


if __name__ == "__main__":
    main()
