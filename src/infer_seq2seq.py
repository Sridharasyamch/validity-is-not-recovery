"""Greedy decoding for the in-distribution supervised corrector.

Emits raw predictions to a JSON file; scoring happens locally with RDKit, since
the GPU host's RDKit build segfaults on invalid input.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
from train_seq2seq import tok, make_model, UNK, PAD, SOS, EOS


@torch.no_grad()
def translate(model, smis, src_stoi, trg_itos, device, batch_size=256,
              max_len=202):
    out = [""] * len(smis)
    order = sorted(range(len(smis)), key=lambda i: len(smis[i]))
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        seqs = []
        for i in idxs:
            try:
                t = tok(smis[i])
            except Exception:
                t = list(smis[i])
            seqs.append([SOS] + [src_stoi.get(x, UNK) for x in t][:max_len - 2] + [EOS])
        L = max(len(x) for x in seqs)
        src = torch.full((len(seqs), L), PAD, dtype=torch.long, device=device)
        for r, ids in enumerate(seqs):
            src[r, :len(ids)] = torch.tensor(ids, device=device)
        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)
        trg = torch.full((len(seqs), 1), SOS, dtype=torch.long, device=device)
        done = torch.zeros(len(seqs), dtype=torch.bool, device=device)
        preds = []
        for _ in range(max_len):
            trg_mask = model.make_trg_mask(trg)
            logits, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
            nxt = logits.argmax(2)[:, -1:]
            preds.append(nxt)
            trg = torch.cat((trg, nxt), 1)
            done |= (nxt.squeeze(1) == EOS)
            if bool(done.all()):
                break
        arr = torch.cat(preds, 1).tolist()
        for r, i in enumerate(idxs):
            toks = []
            for t in arr[r]:
                if t == EOS:
                    break
                toks.append(trg_itos[t])
            out[i] = "".join(toks[::-1])       # target was emitted reversed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/seq2seq/corrector.pt")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--field", default="corrupted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    src_itos, trg_itos = ck["src_itos"], ck["trg_itos"]
    src_stoi = {w: i for i, w in enumerate(src_itos)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(src_itos), len(trg_itos), device)
    model.load_state_dict(ck["model"]); model.eval()

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    preds = translate(model, [r[args.field] for r in rows], src_stoi, trg_itos,
                      device, args.batch_size)
    with open(args.out, "w") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"ground_truth": r.get("ground_truth"),
                                "corrupted": r[args.field],
                                "category": r.get("observed_category"),
                                "pred": p}) + "\n")
    print(f"wrote {args.out} ({len(preds)} predictions)")


if __name__ == "__main__":
    main()
