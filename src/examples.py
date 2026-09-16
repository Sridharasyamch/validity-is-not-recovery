"""Extract qualitative repair examples (invalid -> RDKit diagnosis -> repair)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import symbolic as S

TAG = "Qwen2.5-14B-Instruct__iterative"
rows = [json.loads(l) for l in open(f"results/{TAG}.jsonl")]
good = [r for r in rows if r["valid"] and r["repaired"]]

# pick diverse, clean examples: a couple per category, prefer small edits & short strings
by_cat = {}
for r in good:
    by_cat.setdefault(r["category"], []).append(r)

WANT = ["semantic_valence", "semantic_kekulize", "syntax_ring",
        "syntax_parens", "syntax_brackets", "syntax_other"]
print(f"{'category':18} {'invalid':42} {'diagnosis':46} repair")
print("-" * 150)
picked = []
for cat in WANT:
    cand = sorted(by_cat.get(cat, []),
                  key=lambda r: (len(r["original"]), r.get("token_edit") or 99))
    for r in cand[:2]:
        d = S.diagnose(r["original"])
        # shorten the diagnosis to its informative core
        msg = d.message.split("  ")[0][:44]
        print(f"{cat:18} {r['original'][:42]:42} {msg:46} {r['repaired'][:34]}")
        picked.append({"category": cat, "invalid": r["original"],
                       "diagnosis": d.message, "repair": r["repaired"],
                       "token_edit": r.get("token_edit")})
json.dump(picked, open("paper/examples.json", "w"), indent=2)
print(f"\nsaved {len(picked)} examples -> paper/examples.json")
