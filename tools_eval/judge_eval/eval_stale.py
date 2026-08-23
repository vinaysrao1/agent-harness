"""Measure the stale-failure detector against the labeled corpus."""
import json, sys
sys.path.insert(0, "tools_eval/judge_eval")
from stale_failure import detect

def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return max(0,c-h),min(1,c+h)

recs=[json.loads(l) for l in open("tools_eval/judge_eval/corpus.jsonl")]
fp=[r for r in recs if r["label"]=="false_pass"]
tp=[r for r in recs if r["label"]=="true_pass"]
c=sum(1 for r in fp if detect(r["actions"]))
w=sum(1 for r in tp if detect(r["actions"]))
rl,rh=wilson(c,len(fp)); fl,fh=wilson(w,len(tp))
prec=c/(c+w) if (c+w) else 0
print("STALE-FAILURE DETECTOR (deterministic, no model)\n")
print(f"  recall       {c:3d}/{len(fp):3d} = {c/len(fp):5.1%}  [{rl:.0%},{rh:.0%}]")
print(f"  false-reject {w:3d}/{len(tp):3d} = {w/len(tp):5.1%}  [{fl:.0%},{fh:.0%}]")
print(f"  precision    {prec:5.1%}")
print(f"  BAR (recall>=40%, false-rej<=5%): {'PASS' if c/len(fp)>=0.40 and w/len(tp)<=0.05 else 'FAIL'}")
print("\n  caught:")
for r in fp:
    d=detect(r["actions"])
    if d: print(f"    + {r['task']:<34} {d['commands'][0][:70]}")
print("\n  wrongly flagged:")
for r in tp:
    d=detect(r["actions"])
    if d: print(f"    - {r['task']:<34} {d['commands'][0][:70]}")
