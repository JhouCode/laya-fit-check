# laya-fit-check

A small kit to find out whether [Laya](https://github.com/convaiinnovations/laya), a zero-shot
text classifier, fits **your** data before you build on it.

I tried Laya on my own material and it didn't work for my use case (full write-up:
[English](https://ksmit.com.br/en/blog/mais-ia-pra-que) ·
[Português](https://ksmit.com.br/blog/mais-ia-pra-que)). This kit contains what I
wish I'd had at the start, so you can find out in an afternoon instead of a week. To be clear:
it does not show that Laya is bad. The published benchmark reproduces exactly (see step 1).
What I found is that my tasks are outside the regime that benchmark covers.

## Step 1: reproduce the published number

```
python reproduce_massive.py --device cuda     # or --device cpu
```

This rebuilds the maker's MASSIVE (English intent) benchmark case by case, following
`research/scripts/bench_local.py`. The target is `accuracy 0.7833` on 300 items with 20 options,
and the script exits with code 3 if it misses by more than 0.02. If it fails, don't trust
anything else your install measures.

My result is in `results/massive_en.json`: **0.7833**, an exact match. I got the same number on
three setups: CPU on Linux, CPU on Windows, and ROCm on an RX 9060 XT. The run used laya 0.3.6
and torch 2.9.1.

## Step 2: compare against baselines that cost nothing

```
python compare.py --data mydata.jsonl --task task.json [--rules rules.json] \
                  [--variant english|multilingual] [--device cuda] [--balanced]
```

- `mydata.jsonl`: one `{"text": ..., "label": ..., "group": ...}` per line. The `group` is
  optional, e.g. a document id. When it's there, cross-validation splits by group, so each
  document is always scored by a model that never saw it.
- `task.json`: the question you'd ask Laya, plus one line describing each label.
- `rules.json` (optional): an ordered keyword if/else, the kind you write in an afternoon.
  Write it once and don't tune it against your labels.

`compare.py` prints four numbers per population:

| baseline | what it is |
|---|---|
| majority | always answers the most common label. This is what "no model" scores. |
| rule | your afternoon if/else |
| tfidf | TF-IDF + logistic regression, cross-validated on your labels |
| laya | zero-shot, using your question and label descriptions |

With `--balanced`, it also scores an equal-per-label subset, where majority drops to chance.

**This is not a fair fight, and it isn't meant to be.** TF-IDF learns from your labels; Laya
never sees them. TF-IDF is there as a check that the answer can actually be read from the text.
The fair questions for Laya are: does it beat majority, and does it beat the rule? If it beats
neither, zero-shot isn't buying you anything on this data yet.

`examples/paper_sections/` has the task and rules I used. The corpus itself is not included,
because the paragraphs come from copyrighted papers.

## What I got

The task: which section of a scientific paper a paragraph comes from. There were five labels
(introduction, methods, results, discussion, conclusion), 5,489 English paragraphs from 73
physics papers, and section titles were stripped from the text. I used the English-only variant
with the question in English (`results/paper_sections_en.json`).

| | natural (n=5,489) | balanced (74 per label, n=370) |
|---|---|---|
| majority | 42.4% | 20.0% (chance) |
| rule | 35.5% | 18.1% |
| tfidf, split by paper | **70.5%** | **45.7%** |
| **Laya, english variant, zero-shot** | **25.6%** | **18.9%** |

TF-IDF shows the section can be read from the text, even for papers it never saw. It stayed at
69.8% when each paragraph was cut to 800 characters, less text than Laya reads. The categories
are also not jargon of mine. Even so, Laya scored below majority on the natural population and
at chance on the balanced one.

Two more things worth knowing:

- **The prompt language barely mattered.** I first ran this task with the question and label
  descriptions in Portuguese, and the English variant got 28.1% natural / 21.9% balanced. With
  the multilingual variant and the Portuguese prompt, it got 17.0% balanced. In English it got
  18.9%. My real goal was classifying my own Portuguese notes, and English is not a rescue on
  this task either.
- **Confidence depends on the setup.** With the Portuguese prompt, the mean confidence was 31.5%
  against 28.1% accuracy, which is honest. With the English prompt it was 42.3% against 25.6%.
  On MASSIVE itself it's 96.0% against 78.3%. Measure calibration on your own data before you
  use the confidence as a gate.

About the balanced TF-IDF number: `compare.py` trains on the whole corpus and scores the
balanced subset (45.7%). In my original run, TF-IDF was retrained on the 370 balanced items
alone, with 3 folds split by paper, and got 47.6%.

## Where I'd still try it

The maker's number is for short English utterances with world-level intents. My tasks were long
paragraphs and my own categories. My guess is that the gap is *short utterance vs long passage*
more than *English vs other languages*. That is a hypothesis, not something I measured. If your
data looks like MASSIVE (short messages, intents), step 2 should tell you quickly whether it
works for you.

## Files

- `laya_core.py`: model loading, scoring and metrics. Portions are adapted from
  `convaiinnovations/laya` (`research/scripts/bench_local.py`), Apache-2.0; the file header
  lists the changes.
- `reproduce_massive.py`: step 1.
- `compare.py`: step 2.
- `results/`: my JSON outputs (numbers only, no text).
- `requirements.txt`: install the torch build for your hardware first.

## License

MIT for this kit (see `LICENSE`). Laya itself is Apache-2.0 and is not redistributed here.
