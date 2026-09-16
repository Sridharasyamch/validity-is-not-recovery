"""Character-level LSTM language model over SMILES.

This is the *generator* whose mistakes we study. Char-level decoding (as opposed
to token-level) is the classic, ecologically-valid source of invalid SMILES:
the model can break multi-character atoms ('Cl','Br','[nH]'), drop ring digits,
or unbalance branches -- exactly the "hallucinations" we want to repair.

CLI:
    python gen_model.py train  --data data/guacamol_train.smiles --out runs/lm --limit 250000 --epochs 6
    python gen_model.py sample --ckpt runs/lm/best.pt --n 50000 --temp 1.0 --out gen.smiles
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Vocab + data
# --------------------------------------------------------------------------- #
def build_vocab(smiles: list[str]) -> dict:
    chars = set()
    for s in smiles:
        chars.update(s)
    itos = [PAD, BOS, EOS] + sorted(chars)
    stoi = {c: i for i, c in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


def load_smiles(path: str, limit: int | None = None) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], stoi: dict, max_len: int = 120):
        self.data = [s for s in smiles if len(s) <= max_len]
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        ids = [self.stoi[BOS]] + [self.stoi[c] for c in s] + [self.stoi[EOS]]
        return torch.tensor(ids, dtype=torch.long)


def collate(batch, pad_id: int):
    maxlen = max(len(x) for x in batch)
    out = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : len(x)] = x
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=512, layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, vocab_size)
        self.cfg = dict(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers)

    def forward(self, x, hc=None):
        e = self.emb(x)
        out, hc = self.lstm(e, hc)
        return self.head(out), hc


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args):
    dev = get_device()
    print(f"[train] device={dev}")
    smiles = load_smiles(args.data, args.limit)
    print(f"[train] loaded {len(smiles)} SMILES")
    vocab = build_vocab(smiles)
    stoi, itos = vocab["stoi"], vocab["itos"]
    print(f"[train] vocab size={len(itos)}")

    ds = SmilesDataset(smiles, stoi, args.max_len)
    pad_id = stoi[PAD]
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id), drop_last=True,
                    num_workers=args.workers)

    model = LSTMLM(len(itos), args.emb, args.hidden, args.layers, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vocab, open(os.path.join(args.out, "vocab.json"), "w"))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params/1e6:.1f}M  batches/epoch={len(dl)}")
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for bi, batch in enumerate(dl):
            batch = batch.to(dev)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * batch.size(0)
            seen += batch.size(0)
            if bi % args.log_every == 0:
                print(f"  ep{ep} b{bi}/{len(dl)} loss={loss.item():.3f} "
                      f"({seen/(time.time()-t0):.0f} smi/s)", flush=True)
            if args.max_steps and bi >= args.max_steps:
                break
        avg = tot / max(seen, 1)
        print(f"[train] epoch {ep} avg_loss={avg:.4f} time={time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "cfg": model.cfg, "vocab": vocab},
                   os.path.join(args.out, "last.pt"))
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "cfg": model.cfg, "vocab": vocab},
                       os.path.join(args.out, "best.pt"))
            print(f"[train] saved best (loss={best:.4f})")


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(args):
    dev = get_device()
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    vocab = ckpt["vocab"]
    stoi, itos = vocab["stoi"], vocab["itos"]
    model = LSTMLM(**ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bos, eos, pad = stoi[BOS], stoi[EOS], stoi[PAD]

    out_f = open(args.out, "w")
    n_done = 0
    bs = args.batch
    t0 = time.time()
    skip = {pad, bos}
    while n_done < args.n:
        cur = min(bs, args.n - n_done)
        x = torch.full((cur, 1), bos, dtype=torch.long, device=dev)
        hc = None
        tokens = torch.full((cur, args.max_len), pad, dtype=torch.long, device=dev)
        finished = torch.zeros(cur, dtype=torch.bool, device=dev)
        for t in range(args.max_len):
            logits, hc = model(x, hc)
            logits = logits[:, -1, :] / args.temp
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(1)          # (cur,)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            tokens[:, t] = nxt
            finished = finished | (nxt == eos)
            x = nxt.unsqueeze(1)
            if bool(finished.all()):                              # one sync/step
                break
        rows = tokens.cpu().tolist()                              # one transfer/batch
        for row in rows:
            chars = []
            for tid in row:
                if tid == eos:
                    break
                if tid not in skip:
                    chars.append(itos[tid])
            out_f.write("".join(chars) + "\n")
        n_done += cur
        print(f"[sample] {n_done}/{args.n} ({n_done/(time.time()-t0):.0f} smi/s)", flush=True)
    out_f.close()
    print(f"[sample] wrote {n_done} samples to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="runs/lm")
    t.add_argument("--limit", type=int, default=250000)
    t.add_argument("--max_len", type=int, default=120)
    t.add_argument("--emb", type=int, default=128)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--max_steps", type=int, default=0)
    t.set_defaults(func=train)

    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--n", type=int, default=50000)
    s.add_argument("--temp", type=float, default=1.0)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--max_len", type=int, default=120)
    s.add_argument("--out", default="gen.smiles")
    s.set_defaults(func=sample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
