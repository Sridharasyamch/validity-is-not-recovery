# Validity Is Not Recovery

Benchmark, code and per-item results for *"Validity Is Not Recovery: Constrained
Candidate Selection for Molecular String Repair."*

Generative models for molecules frequently emit SMILES strings that no chemistry
toolkit will parse, and a growing literature aims to correct them. That
literature is evaluated almost entirely by **validity** — the fraction of
repaired strings that parse. This work shows validity is a weak proxy for
whether the *intended* molecule is recovered, and that validity-only evaluation
can substantially mis-rank recovery systems.

On a leakage-controlled benchmark of 6,000 corrupted SMILES with known ground
truth, a recently published corrector reaches 86.6% validity but recovers the
intended molecule for 1.2% of inputs — roughly 74 valid molecules per correct
one. Reformulating repair as constrained candidate *selection* rather than
sequence generation gives 100.0% validity and 80.7% exact recovery.

## Method

Six stages, of which only one is learned:

1. **Diagnose** – RDKit parse, then sanitise, yielding an error category.
2. **Enumerate** – every string one token edit from the input.
3. **Verify** – keep those RDKit accepts; the surviving set is sound by construction.
4. **Prune** – drop edits inconsistent with the diagnosis (lossless on dev).
5. **Rank** – score survivors by chemical-language-model likelihood.
6. **Recover or abstain** – return the top candidate, or decline when unsure.

## Layout

```
src/                     all code (see "Reproducing" below)
data/recovery/           the 6,000-item benchmark (dev 1,800 / test 4,200)
data/recovery_multi/     k = 1, 2, 3 simultaneous-edit benchmarks
data/recovery_external/  benchmark built with UnCorrupt's operators on PAPYRUS
data/recovery_gendist*/  corruptions of each generator's own outputs
data/benchmark/          real invalid SMILES harvested from a char-level generator
results_recovery/        per-item outputs and summaries for every system
paper_vnr/               manuscript source
```

## Reproducing

```bash
pip install -r requirements.txt

# every headline number in the paper, checked against the raw result files
python src/verify_paper_numbers.py        # 135 assertions

# rebuild the candidate sets (excluded from git; ~195 MB, deterministic)
python src/enumerate_repairs.py --benchmark data/recovery/recovery_test.jsonl \
                                --out data/recovery/cands_test.jsonl

# rank them
python src/rank_domain_lm.py --cands data/recovery/cands_test.jsonl \
                             --mode mean --prune --tag TEST_rerun
```

`verify_paper_numbers.py` is the gate: it asserts every number in the manuscript
against the files in `results_recovery/`, and fails loudly rather than reporting
a mismatch. It also guards the build (the IEEE variant must not drift from the
main source) and the prose.

## Not in this repository

Excluded to keep the tree small; each is one command or one download away.

| | Why | How to obtain |
|---|---|---|
| GuacaMol corpora | publicly distributed | [GuacaMol](https://github.com/BenevolentAI/guacamol) |
| Candidate-set caches (~195 MB) | deterministic | `src/enumerate_repairs.py` |
| Supervised-training corpus (71 MB) | deterministic | `src/corrupt.py --n 250000 --seed 777` |
| Model checkpoints (~128 MB) | size | published as release assets |
| SmiSelf / UnCorrupt baselines | third-party | clone into `third_party/`, then set `SMISELF_PATH` / `UNCORRUPT_PATH` |

## Scope

This is a precise repair mechanism, not a general molecule reconstructor. It is
near-complete for single-token corruptions (99.79% candidate coverage) and of
almost no use when two errors coincide (0.50%). Roughly 29% of the external
benchmark lies beyond edit-bounded enumeration altogether. Where the method
cannot answer, it returns the input unchanged rather than guessing.

## Licence

MIT. See `LICENSE`.
