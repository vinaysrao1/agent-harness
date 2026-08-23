"""Sweep post-hoc filters over recorded verdicts. No new API calls.

A flag only counts when the judge said not_verified AND the filter admits it.
Filters are applied to the FLAG decision only: suppressing a flag turns a
would-be rejection into an acceptance, so it can only ever trade recall for
false-rejection - which is exactly the trade we need, since false-rejection is
what fails the bar.
"""
from __future__ import annotations
import json, sys

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return max(0.0, c-h), min(1.0, c+h)

def score(rows, admit):
    fp = [r for r in rows if r["label"] == "false_pass"]
    tp = [r for r in rows if r["label"] == "true_pass"]
    flag = lambda r: r["verified"] is False and admit(r)
    c = sum(1 for r in fp if flag(r)); w = sum(1 for r in tp if flag(r))
    prec = c/(c+w) if (c+w) else 0.0
    return c, len(fp), w, len(tp), prec

def line(name, rows, admit):
    c, nfp, w, ntp, prec = score(rows, admit)
    rl, rh = wilson(c, nfp); fl, fh = wilson(w, ntp)
    ok = (c/nfp) >= 0.40 and (w/ntp) <= 0.05
    print(f"  {name:38s} recall {c:2d}/{nfp} = {c/nfp:5.1%} [{rl:.0%},{rh:.0%}]   "
          f"false-rej {w:2d}/{ntp} = {w/ntp:5.1%} [{fl:.0%},{fh:.0%}]   "
          f"prec {prec:5.1%}  {'** PASS **' if ok else ''}")

cache = json.load(open("tools_eval/judge_eval/verdicts.json"))
for cfg in sys.argv[1:]:
    rows = [v for k, v in cache.items() if k.startswith(cfg + "::")]
    if not rows:
        print(f"{cfg}: no verdicts"); continue
    has_conf = any(r.get("confidence") is not None for r in rows)
    print(f"\n{'='*104}\n{cfg}  (n={len(rows)})\n{'='*104}")
    line("no filter (baseline)", rows, lambda r: True)
    if not has_conf:
        continue
    for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        line(f"confidence >= {t}", rows, lambda r, t=t: (r.get("confidence") or 0) >= t)
    line("evidence_sufficient only", rows, lambda r: bool(r.get("evidence_sufficient")))
    for t in (0.7, 0.8):
        line(f"evidence_sufficient AND conf >= {t}", rows,
             lambda r, t=t: bool(r.get("evidence_sufficient")) and (r.get("confidence") or 0) >= t)
