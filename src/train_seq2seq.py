"""Supervised seq2seq corrector baseline, trained in-distribution.

This is the obvious alternative to our approach and the strongest baseline in the
literature's family: the UnCorrupt SMILES transformer architecture (Schoenmaker et
al., J. Cheminformatics 2023), retrained from scratch on OUR corruption
distribution so it suffers no domain shift. It sees 245k (corrupted -> correct)
pairs built from GuacaMol train molecules, disjoint from every benchmark
molecule, and is therefore given every advantage the benchmark allows.

Target tokens are emitted in reverse, matching the original implementation.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn

SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK, PAD, SOS, EOS = 0, 1, 2, 3


def tok(smi, reverse=False):
    import re
    pattern = (r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+"
               r"|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
    t = re.findall(pattern, smi)
    return t[::-1] if reverse else t


def build_vocab(pairs):
    cs, ct = Counter(), Counter()
    for a, b in pairs:
        cs.update(tok(a)); ct.update(tok(b, reverse=True))
    def mk(c):
        itos = list(SPECIALS) + [w for w, _ in c.most_common()]
        return itos, {w: i for i, w in enumerate(itos)}
    return mk(cs), mk(ct)


def encode(pairs, src_stoi, trg_stoi, max_len=200):
    out = []
    for a, b in pairs:
        s = [SOS] + [src_stoi.get(t, UNK) for t in tok(a)][:max_len] + [EOS]
        t_ = [SOS] + [trg_stoi.get(t, UNK) for t in tok(b, reverse=True)][:max_len] + [EOS]
        out.append((s, t_))
    return out


def batches(data, bs, shuffle=True, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    # length-bucketed for padding efficiency
    idx.sort(key=lambda i: len(data[i][0]) // 8)
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        random.Random(seed + 1).shuffle(chunks)
    for ch in chunks:
        ss = [data[i][0] for i in ch]; tt = [data[i][1] for i in ch]
        Ls, Lt = max(len(x) for x in ss), max(len(x) for x in tt)
        src = torch.full((len(ch), Ls), PAD, dtype=torch.long)
        trg = torch.full((len(ch), Lt), PAD, dtype=torch.long)
        for r, (a, b) in enumerate(zip(ss, tt)):
            src[r, :len(a)] = torch.tensor(a); trg[r, :len(b)] = torch.tensor(b)
        yield src, trg


def make_model(n_src, n_trg, device, layers=3, hid=256, heads=8, pf=512,
               dropout=0.1, max_len=202):
    up = os.environ.get("UNCORRUPT_PATH", "third_party/uncorrupt")
    if up not in sys.path:
        sys.path.insert(0, up)
    from src.transformer import Encoder, Decoder, Seq2Seq
    enc = Encoder(n_src, hid, layers, heads, pf, dropout, max_len, device)
    dec = Decoder(n_trg, hid, layers, heads, pf, dropout, max_len, device)
    return Seq2Seq(enc, dec, PAD, PAD, device, loader_train=None, out="").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/recovery_train/train.jsonl")
    ap.add_argument("--val", default="data/recovery_train/val.jsonl")
    ap.add_argument("--out", default="runs/seq2seq/corrector.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=90.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def load(p):
        out = []
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            out.append((r["corrupted"], r["ground_truth"]))
        return out

    tr, va = load(args.train), load(args.val)
    if args.limit:
        tr = tr[:args.limit]
    (src_itos, src_stoi), (trg_itos, trg_stoi) = build_vocab(tr)
    print(f"[s2s] train={len(tr)} val={len(va)} src_vocab={len(src_itos)} "
          f"trg_vocab={len(trg_itos)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(src_itos), len(trg_itos), device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[s2s] device={device} params={n_par/1e6:.2f}M", flush=True)

    dtr = encode(tr, src_stoi, trg_stoi)
    dva = encode(va, src_stoi, trg_stoi)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    t0 = time.time()
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for src, trg in batches(dtr, args.bs, seed=ep):
            src, trg = src.to(device), trg.to(device)
            opt.zero_grad()
            out, _ = model(src, trg[:, :-1])
            loss = crit(out.reshape(-1, out.size(-1)), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for src, trg in batches(dva, args.bs, shuffle=False):
                src, trg = src.to(device), trg.to(device)
                out, _ = model(src, trg[:, :-1])
                vt += crit(out.reshape(-1, out.size(-1)),
                           trg[:, 1:].reshape(-1)).item(); vn += 1
        mins = (time.time() - t0) / 60
        vl = vt / max(vn, 1)
        print(f"[s2s] epoch {ep} train={tot/max(n,1):.4f} val={vl:.4f} "
              f"ppl={math.exp(min(vl,20)):.2f} ({mins:.1f} min)", flush=True)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(),
                        "src_itos": src_itos, "trg_itos": trg_itos,
                        "cfg": {"layers": 3, "hid": 256, "heads": 8, "pf": 512}},
                       args.out)
            print(f"[s2s]   saved (val {vl:.4f})", flush=True)
        if mins > args.max_minutes:
            print("[s2s] time budget reached", flush=True); break
    print(f"[s2s] done, best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
