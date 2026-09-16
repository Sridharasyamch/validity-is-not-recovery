"""UnCorrupt SMILES (Schoenmaker et al., J. Cheminformatics 2023) as a recovery baseline.

The published repository ships trained weights but drives them through
`torchtext.legacy`, which no longer installs on modern PyTorch. We therefore load
the released checkpoint directly and rebuild the vocabulary by replicating
torchtext's `Field.build_vocab` ordering exactly (specials first, then token
types sorted alphabetically and stable-sorted by descending frequency over the
released error corpus, which is the same corpus the authors' script builds the
vocabulary from -- their split uses frac_test=0, so train+valid is the whole
file). Decoding replicates `Seq2Seq.translate`: greedy argmax, reversed target
tokenisation, truncation at the first <eos>.

`--selftest` runs the model on the authors' own corruption corpus. If the
checkpoint and vocabulary are wired up correctly this reproduces the behaviour
reported in the paper; if the vocabulary order were wrong the output would be
noise. We verify that before reporting any number on our benchmark.
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import torch

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import recovery_metrics as RM

UNCORRUPT = os.environ.get("UNCORRUPT_PATH", "third_party/uncorrupt")
CKPT = f"{UNCORRUPT}/Data/performance/transformer_multiple_12_PAPYRUS_200_16_3.pkg"
ERRORS_CSV = f"{UNCORRUPT}/Data/errors/PAPYRUS_200_multiple_12_errors.csv"

SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


def _smi_tokenizer(smi, reverse=False):
    import re
    pattern = (r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+"
               r"|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
    toks = re.findall(pattern, smi)
    return toks[::-1] if reverse else toks


def _build_vocab(counter):
    """Replicate torchtext.legacy.vocab.Vocab ordering."""
    c = Counter(counter)
    for s in SPECIALS:
        del c[s]
    itos = list(SPECIALS)
    wf = sorted(c.items(), key=lambda t: t[0])          # alphabetical
    wf.sort(key=lambda t: t[1], reverse=True)           # stable, by freq desc
    itos += [w for w, _ in wf]
    return itos, {w: i for i, w in enumerate(itos)}


def load_fields():
    src_c, trg_c = Counter(), Counter()
    with open(ERRORS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            err, std = row.get("ERROR"), row.get("STD_SMILES")
            if not err or not std:
                continue
            try:
                src_c.update(_smi_tokenizer(err))
                trg_c.update(_smi_tokenizer(std, reverse=True))
            except Exception:
                continue
    src_itos, src_stoi = _build_vocab(src_c)
    trg_itos, trg_stoi = _build_vocab(trg_c)
    return (src_itos, src_stoi), (trg_itos, trg_stoi)


def build_model(n_src, n_trg, device, layers=3, hid=256, heads=8, pf=512,
                max_len=202):
    if UNCORRUPT not in sys.path:
        sys.path.insert(0, UNCORRUPT)
    from src.transformer import Encoder, Decoder, Seq2Seq
    enc = Encoder(n_src, hid, layers, heads, pf, 0.1, max_len, device)
    dec = Decoder(n_trg, hid, layers, heads, pf, 0.1, max_len, device)
    m = Seq2Seq(enc, dec, SPECIALS.index("<pad>"), SPECIALS.index("<pad>"),
                device, loader_train=None, out="", TRG=None, SRC=None)
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    m.load_state_dict(sd)
    return m.to(device).eval()


@torch.no_grad()
def translate(model, smis, src_stoi, trg_itos, device, batch_size=128,
              max_len=202):
    """Greedy decode, faithful to Seq2Seq.translate + is_smiles(reverse=True)."""
    unk, pad, sos, eos = 0, 1, 2, 3
    out = [""] * len(smis)
    order = sorted(range(len(smis)), key=lambda i: len(smis[i]))
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        seqs = []
        for i in idxs:
            try:
                toks = _smi_tokenizer(smis[i])
            except Exception:
                toks = list(smis[i])
            ids = [sos] + [src_stoi.get(t, unk) for t in toks][:max_len - 2] + [eos]
            seqs.append(ids)
        L = max(len(x) for x in seqs)
        src = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
        for r, ids in enumerate(seqs):
            src[r, :len(ids)] = torch.tensor(ids, device=device)

        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)
        trg = torch.full((len(seqs), 1), sos, dtype=torch.long, device=device)
        done = torch.zeros(len(seqs), dtype=torch.bool, device=device)
        preds = []
        for _ in range(max_len):
            trg_mask = model.make_trg_mask(trg)
            logits, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
            nxt = logits.argmax(2)[:, -1:]
            preds.append(nxt)
            trg = torch.cat((trg, nxt), 1)
            done |= (nxt.squeeze(1) == eos)
            if bool(done.all()):
                break                       # every row already emitted <eos>
        arr = torch.cat(preds, 1).tolist()  # [batch, steps]
        for r, i in enumerate(idxs):
            toks = [trg_itos[t] for t in arr[r]]
            rev = list(itertools.takewhile(lambda x: x != "<eos>", toks))
            out[i] = "".join(rev[::-1])     # target was tokenised reversed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark")
    ap.add_argument("--selftest", type=int, default=0,
                    help="run on N rows of the authors' own corruption corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--tag", default="uncorrupt")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    (src_itos, src_stoi), (trg_itos, trg_stoi) = load_fields()
    print(f"[uncorrupt] rebuilt vocab: SRC={len(src_itos)} TRG={len(trg_itos)} "
          f"(checkpoint expects 54 / 53)")
    model = build_model(len(src_itos), len(trg_itos), device)
    print("[uncorrupt] checkpoint loaded")

    if args.selftest:
        rows = []
        with open(ERRORS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("ERROR") and row.get("STD_SMILES"):
                    rows.append((row["ERROR"], row["STD_SMILES"]))
                if len(rows) >= args.selftest:
                    break
        preds = translate(model, [r[0] for r in rows], src_stoi, trg_itos,
                          device, args.batch_size)
        scored = [RM.score(t, p, with_mcs=False) for (_, t), p in zip(rows, preds)]
        agg = RM.aggregate(scored)
        print("[uncorrupt] SELF-TEST on the authors' own corruption corpus "
              f"(n={len(rows)}):")
        print(json.dumps({k: agg[k] for k in
                          ("validity", "exact_recovery", "tanimoto_given_valid")},
                         indent=2))
        for (e, t), p in list(zip(rows, preds))[:3]:
            print(f"   in : {e}\n   out: {p}\n   trg: {t}\n")
        return

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    preds = translate(model, [r["corrupted"] for r in rows], src_stoi, trg_itos,
                      device, args.batch_size)

    recs = []
    for i, (r, p) in enumerate(zip(rows, preds)):
        sc = RM.score(r["ground_truth"], p, with_mcs=(i < args.mcs_n))
        recs.append({"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                     "category": r["observed_category"], "pred": p, **sc})
    agg = RM.aggregate(recs)
    agg["method"] = "uncorrupt"
    cats = {}
    for c in sorted(set(r["category"] for r in recs)):
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
