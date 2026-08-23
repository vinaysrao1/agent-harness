"""Report judge performance with confidence intervals, separating the records
used to pick the configuration from the ones held out.

The calibrated framing was chosen after seeing results on the first 10
false_pass + 10 true_pass records, so those 20 are contaminated by selection.
The held-out numbers are the honest estimate.
"""
from __future__ import annotations
import json, sys
from math import comb

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))

corpus = [json.loads(l) for l in open("tools_eval/judge_eval/corpus.jsonl")]
screened = set()
nf = nt = 0
for r in corpus:  # same order --limit 10 used
    if r["label"] == "false_pass" and nf < 10:
        screened.add((r["run"], r["task"])); nf += 1
    elif r["label"] == "true_pass" and nt < 10:
        screened.add((r["run"], r["task"])); nt += 1

cache = json.load(open("tools_eval/judge_eval/verdicts.json"))
by_cfg: dict[str, list[dict]] = {}
for k, v in cache.items():
    by_cfg.setdefault(k.split("::")[0], []).append(v)

for cfg in sys.argv[1:]:
    rows = by_cfg.get(cfg, [])
    if not rows:
        print(f"{cfg}: no verdicts"); continue
    print(f"\n{'='*70}\n{cfg}   ({len(rows)} verdicts)")
    for scope in ("ALL", "HELD-OUT"):
        sel = rows if scope == "ALL" else [
            r for r in rows if (r["run"], r["task"]) not in screened]
        fp = [r for r in sel if r["label"] == "false_pass"]
        tp = [r for r in sel if r["label"] == "true_pass"]
        if not fp or not tp: continue
        caught = sum(1 for r in fp if r["verified"] is False)
        wrong = sum(1 for r in tp if r["verified"] is False)
        unp = sum(1 for r in sel if r["verified"] is None)
        rlo, rhi = wilson(caught, len(fp))
        flo, fhi = wilson(wrong, len(tp))
        ships = (caught/len(fp)) >= 0.40 and (wrong/len(tp)) <= 0.05
        print(f"  {scope:9s} recall {caught:3d}/{len(fp):3d} = {caught/len(fp):5.1%} "
              f"[{rlo:.0%},{rhi:.0%}]   "
              f"false-rej {wrong:3d}/{len(tp):3d} = {wrong/len(tp):5.1%} "
              f"[{flo:.0%},{fhi:.0%}]   unparsed {unp}   "
              f"{'PASS' if ships else 'FAIL'}")
