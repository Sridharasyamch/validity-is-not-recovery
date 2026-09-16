"""Decoder-only SMILES Transformer: a second model family for the 2x2 study.

This is Generator B and Ranker B. It deliberately reuses the character
vocabulary of the LSTM generator verbatim, so the two rankers differ ONLY in
architecture -- otherwise an apparent architecture effect could just be a
tokenisation effect.

Modes:
  train   fit on a SMILES corpus (GPU)
  sample  autoregressive sampling, to produce Generator B's own outputs (GPU)
  score   total and mean per-character log-likelihood (CPU is fine)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class SmilesTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, layers=4,
                 dim_ff=1024, dropout=0.1, max_len=160):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.cfg = dict(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                        layers=layers, dim_ff=dim_ff, max_len=max_len)

    def forward(self, x):
        L = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool),
                          diagonal=1)
        h = self.enc(h, mask=mask, src_key_padding_mask=(x == PAD))
        return self.head(self.ln(h))


def load_vocab(vocab_json=None, smiles=None):
    if vocab_json and os.path.exists(vocab_json):
        v = json.load(open(vocab_json))
        return v["itos"], v["stoi"]
    chars = sorted({c for s in smiles for c in s})
    itos = ["<pad>", "<bos>", "<eos>"] + chars
    return itos, {c: i for i, c in enumerate(itos)}


def encode(smis, stoi, max_len):
    out = []
    for s in smis:
        ids = [BOS] + [stoi[c] for c in s if c in stoi][:max_len - 2] + [EOS]
        out.append(ids)
    return out


def batches(data, bs, shuffle=True, seed=0):
    import random
    idx = list(range(len(data)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    idx.sort(key=lambda i: len(data[i]) // 8)
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        random.Random(seed + 1).shuffle(chunks)
    for ch in chunks:
        seqs = [data[i] for i in ch]
        L = max(len(x) for x in seqs)
        t = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for r, q in enumerate(seqs):
            t[r, :len(q)] = torch.tensor(q)
        yield t


def build(cfg, device):
    m = SmilesTransformerLM(cfg["vocab_size"], d_model=cfg["d_model"],
                            nhead=cfg["nhead"], layers=cfg["layers"],
                            dim_ff=cfg["dim_ff"], max_len=cfg["max_len"])
    return m.to(device)


@torch.no_grad()
def score(model, stoi, smis, device="cpu", batch_size=256, max_len=160):
    """Total and mean per-character log-likelihood -- same contract as the LSTM."""
    model.eval()
    out = []
    for s in range(0, len(smis), batch_size):
        chunk = smis[s:s + batch_size]
        seqs = encode(chunk, stoi, max_len)
        L = max(len(q) for q in seqs)
        inp = torch.full((len(seqs), L), PAD, dtype=torch.long, device=device)
        for r, q in enumerate(seqs):
            inp[r, :len(q)] = torch.tensor(q, device=device)
        logits = model(inp[:, :-1])
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt = inp[:, 1:]
        tok = lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        mask = (tgt != PAD).float()
        tot = (tok * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1)
        out.extend([(t, t / c) for t, c in zip(tot.tolist(), cnt.tolist())])
    return out


@torch.no_grad()
def sample(model, itos, n, device, temperature=1.0, max_len=160, batch=512):
    model.eval()
    got = []
    while len(got) < n:
        b = min(batch, n - len(got))
        seq = torch.full((b, 1), BOS, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = model(seq)[:, -1, :] / max(temperature, 1e-6)
            logits[:, PAD] = -1e9
            logits[:, BOS] = -1e9
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            nxt[done] = EOS
            seq = torch.cat([seq, nxt], 1)
            done |= (nxt.squeeze(1) == EOS)
            if bool(done.all()):
                break
        for row in seq.tolist():
            chars = []
            for t in row[1:]:
                if t == EOS:
                    break
                chars.append(itos[t])
            got.append("".join(chars))
    return got[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "sample"])
    ap.add_argument("--data", default="data/guacamol_train.smiles")
    ap.add_argument("--vocab_json", default="runs/lm/vocab.json")
    ap.add_argument("--out", default="runs/tlm/best.pt")
    ap.add_argument("--ckpt", default="runs/tlm/best.pt")
    ap.add_argument("--limit", type=int, default=600000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_minutes", type=float, default=35.0)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample_out", default="runs/tlm/gen_t1.0.smiles")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        smis = []
        with open(args.data) as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s and len(s) <= args.max_len - 2:
                    smis.append(s)
                if len(smis) >= args.limit:
                    break
        itos, stoi = load_vocab(args.vocab_json, smis)
        print(f"[tlm] train={len(smis)} vocab={len(itos)} device={device}", flush=True)
        n_val = 5000
        val, tr = smis[:n_val], smis[n_val:]
        dtr, dva = encode(tr, stoi, args.max_len), encode(val, stoi, args.max_len)
        cfg = dict(vocab_size=len(itos), d_model=256, nhead=8, layers=4,
                   dim_ff=1024, max_len=args.max_len)
        model = build(cfg, device)
        print(f"[tlm] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
              flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        crit = nn.CrossEntropyLoss(ignore_index=PAD)
        t0, best = time.time(), float("inf")
        for ep in range(1, args.epochs + 1):
            model.train(); tot = k = 0
            for t in batches(dtr, args.bs, seed=ep):
                t = t.to(device)
                opt.zero_grad()
                logits = model(t[:, :-1])
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            t[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); k += 1
            model.eval(); vt = vk = 0
            with torch.no_grad():
                for t in batches(dva, args.bs, shuffle=False):
                    t = t.to(device)
                    logits = model(t[:, :-1])
                    vt += crit(logits.reshape(-1, logits.size(-1)),
                               t[:, 1:].reshape(-1)).item(); vk += 1
            vl = vt / max(vk, 1)
            mins = (time.time() - t0) / 60
            print(f"[tlm] epoch {ep} train={tot/max(k,1):.4f} val={vl:.4f} "
                  f"ppl={math.exp(min(vl,20)):.3f} ({mins:.1f} min)", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "vocab": {"itos": itos, "stoi": stoi}}, args.out)
                print(f"[tlm]   saved (val {vl:.4f})", flush=True)
            if mins > args.max_minutes:
                print("[tlm] time budget reached", flush=True); break
        print(f"[tlm] done, best val {best:.4f}", flush=True)
    else:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        itos = ck["vocab"]["itos"]
        model = build(ck["cfg"], device)
        model.load_state_dict(ck["model"])
        os.makedirs(os.path.dirname(args.sample_out), exist_ok=True)
        smis = sample(model, itos, args.n, device, args.temperature,
                      ck["cfg"]["max_len"])
        with open(args.sample_out, "w") as f:
            f.write("\n".join(smis) + "\n")
        print(f"[tlm] wrote {len(smis)} samples -> {args.sample_out}")


if __name__ == "__main__":
    main()
