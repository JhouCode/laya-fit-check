#!/usr/bin/env python3
"""Step 2: run Laya on YOUR data next to three baselines that cost nothing.

    majority      always answer the most common label (what "no model" scores)
    rule          an ordered keyword if/else you write in an afternoon (optional, --rules)
    tfidf         TF-IDF + logistic regression, cross-validated. With a `group` field
                  (e.g. the document id), folds split by group, so it is always tested on
                  documents it never saw.

Laya is zero-shot and the TF-IDF baseline learns from your labels: that is not a fair fight, and
the table says so. The fair questions are: does Laya beat majority? Does it beat the rule?

Data (JSONL, one per line):  {"text": "...", "label": "methods", "group": "paper-17"}
Task (JSON):  {"field": "passage",
               "question": "Which section of a scientific paper was `passage` taken from?",
               "labels": {"methods": "how the work was done: ...", ...}}
Rules (JSON, optional):  {"default": "introduction", "rules": [["conclusion", "\\\\bin conclusion\\\\b"], ...]}
               first match wins; write it once, don't tune it against the labels.

    python compare.py --data mydata.jsonl --task task.json [--rules rules.json]
                      [--variant english|multilingual] [--device cuda] [--balanced]
"""
import argparse, json, os, random, re, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

SEED = 13


def balanced_subset(labels, rng):
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(y, []).append(i)
    n = min(len(v) for v in by.values())
    return sorted(i for y in sorted(by) for i in rng.sample(by[y], n)), n


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def run_rule(texts, spec):
    rules = [(lab, re.compile(rx, re.I)) for lab, rx in spec["rules"]]
    return [next((lab for lab, rx in rules if rx.search(t)), spec["default"]) for t in texts]


def run_tfidf(texts, labels, groups):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, random_state=SEED))
    if groups and len(set(groups)) >= 5:
        return list(cross_val_predict(model, texts, labels, cv=GroupKFold(n_splits=5), groups=groups))
    return list(cross_val_predict(model, texts, labels,
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)))


def run_laya(texts, task, variant, device):
    from laya_core import load, score_cases, softmax_t, temp_for
    names = list(task["labels"])
    q = {"answer": {"type": "choice", "instructions": task["question"], "criteria": task["labels"]}}
    agent = load(variant, device)
    logits, index, secs, dropped = score_cases(agent, [({task["field"]: t}, q) for t in texts])
    pred, conf = [None] * len(texts), [None] * len(texts)
    for (ci, _, qt, k), z in zip(index, logits):
        if z is not None:
            p = softmax_t(z, temp_for(agent, qt, k))
            pred[ci], conf[ci] = names[int(np.argmax(p))], float(np.max(p))
    return pred, conf, secs, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--rules")
    ap.add_argument("--variant", default="english", choices=["english", "multilingual"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--balanced", action="store_true", help="also score an equal-per-label subset")
    ap.add_argument("--skip-laya", action="store_true")
    ap.add_argument("--out", default="results/compare.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    task = json.load(open(a.task, encoding="utf-8"))
    texts = [r["text"] for r in rows]
    gold = [r["label"] for r in rows]
    groups = [r["group"] for r in rows] if all("group" in r for r in rows) else None
    unknown = set(gold) - set(task["labels"])
    if unknown:
        sys.exit("labels in data but not in task.json: %s" % sorted(unknown))

    preds = {"majority": [Counter(gold).most_common(1)[0][0]] * len(gold)}
    if a.rules:
        preds["rule"] = run_rule(texts, json.load(open(a.rules, encoding="utf-8")))
    preds["tfidf (learns from your labels)"] = run_tfidf(texts, gold, groups)
    extra = {}
    if not a.skip_laya:
        lp, lc, secs, dropped = run_laya(texts, task, a.variant, a.device)
        preds["laya %s (zero-shot)" % a.variant] = lp
        ok = [i for i, p in enumerate(lp) if p is not None]
        extra = {"laya_seconds": round(secs, 1), "laya_dropped": dropped,
                 "laya_mean_confidence": round(float(np.mean([lc[i] for i in ok])), 4) if ok else None}

    pops = [("natural", list(range(len(rows))))]
    if a.balanced:
        sel, n = balanced_subset(gold, random.Random(SEED))
        pops.append(("balanced (%d per label)" % n, sel))
    report = {"n": len(rows), "labels": dict(Counter(gold)), "grouped_cv": bool(groups), **extra, "results": {}}
    for pname, idx in pops:
        print("\n%s · n=%d" % (pname, len(idx)))
        report["results"][pname] = {}
        for name, p in preds.items():
            if name == "majority" and pname != "natural":
                acc = 1 / len(set(gold[i] for i in idx))   # balanced: majority = chance
            else:
                ii = [i for i in idx if p[i] is not None]
                acc = accuracy([gold[i] for i in ii], [p[i] for i in ii])
            report["results"][pname][name] = round(acc, 4)
            print("  %-34s %6.1f%%" % (name, 100 * acc))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
