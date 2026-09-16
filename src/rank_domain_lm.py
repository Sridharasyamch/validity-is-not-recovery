"""Rank enumerated repairs with a DOMAIN chemical LM (contamination control).

The general-purpose LLM used for ranking was pretrained on web text that almost
certainly contains public molecule databases, so a sceptical reader can ask
whether its ranking reflects a chemical prior or memorisation of the specific
test molecules. This ranker answers that: it is the character-level SMILES LSTM
we trained ourselves on the GuacaMol TRAINING split only, which is disjoint from
every molecule in the benchmark (verified: zero canonical overlap). If it also
beats uniform selection by a wide margin, the effect is a genuine chemical prior
rather than test-set memorisation.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import recovery_metrics as RM
import edit_ops as EO
from gen_model import LSTMLM


def load_lm(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, vocab = ck["cfg"], ck["vocab"]
    m = LSTMLM(cfg["vocab_size"], emb=cfg["emb"], hidden=cfg["hidden"],
               layers=cfg["layers"])
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    stoi = vocab["stoi"]
    return m, stoi


@torch.no_grad()
def score(model, stoi, smis, device="cpu", batch_size=256):
    """Total and mean per-character log-likelihood of each SMILES."""
    bos = stoi.get("<bos>", stoi.get("^", 1))
    eos = stoi.get("<eos>", stoi.get("$", 2))
    unk = stoi.get("<unk>", 0)
    out = []
    for s in range(0, len(smis), batch_size):
        chunk = smis[s:s + batch_size]
        seqs = [[bos] + [stoi.get(c, unk) for c in x] + [eos] for x in chunk]
        L = max(len(q) for q in seqs)
        inp = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
        for r, q in enumerate(seqs):
            inp[r, :len(q)] = torch.tensor(q, device=device)
        logits, _ = model(inp[:, :-1])
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt = inp[:, 1:]
        tok = lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        mask = (tgt != 0).float()
        tot = (tok * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1)
        for t, c in zip(tot.tolist(), cnt.tolist()):
            out.append((t, t / c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--lm", default="runs/lm/best.pt")
    ap.add_argument("--mode", default="sum", choices=["sum", "mean"])
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--max_cands", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--tag", default="enumrank_domainlm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.cands) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(2)                 # keep the machine responsive
    model, stoi = load_lm(args.lm)
    print(f"[domainlm] {len(rows)} items", flush=True)

    j = 0 if args.mode == "sum" else 1
    recs = []
    for i, r in enumerate(rows):
        cands = r["candidates"]
        if args.prune and cands:
            cands, _ = EO.prune(r["corrupted"], cands, r["witness"], r.get("category"))
        best, margin = None, None
        if cands:
            if len(cands) > args.max_cands:
                cands = sorted(cands, key=len)[:args.max_cands]
            sc = score(model, stoi, cands, batch_size=args.batch_size)
            order = sorted(range(len(cands)), key=lambda t: -sc[t][j])
            best = cands[order[0]]
            if len(order) > 1:
                margin = sc[order[0]][j] - sc[order[1]][j]
        s = RM.score(r["ground_truth"], best, with_mcs=(i < args.mcs_n))
        recs.append({"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                     "category": r.get("category"), "pred": best,
                     "n_cands": len(cands), "margin": margin,
                     "truth_in_candidates": r.get("truth_in_candidates"), **s})
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    agg = RM.aggregate(recs)
    agg.update({"ranker": "domain_char_lstm", "mode": args.mode, "prune": args.prune})
    cats = {}
    for c in sorted({r["category"] for r in recs if r["category"]}):
        sub = [r for r in recs if r["category"] == c]
        a = RM.aggregate(sub)
        cats[c] = {"n": a["n"], "validity": a["validity"],
                   "exact_recovery": a["exact_recovery"],
                   "tanimoto_given_valid": a["tanimoto_given_valid"]}
    agg["per_category"] = cats
    with open(os.path.join(args.out_dir, f"{args.tag}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}.summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
