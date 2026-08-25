from __future__ import annotations
import argparse, json, re
from pathlib import Path


def words(s): return re.findall(r"\b\w+\b", s.lower())
def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def wer(ref,hyp):
    r=words(ref); h=words(hyp); return edit_distance(r,h)/max(1,len(r))

def pooled_wer(refs, hyps):
    total_edits = 0
    total_ref_words = 0
    for ref, hyp in zip(refs, hyps):
        r = words(ref); h = words(hyp)
        total_edits += edit_distance(r, h)
        total_ref_words += len(r)
    return total_edits / max(1, total_ref_words)

def main():
    p=argparse.ArgumentParser(description="Compute WER delta from precomputed raw/enhanced transcripts.")
    p.add_argument("--raw-transcripts", required=True, help="JSONL: {id,text}")
    p.add_argument("--enhanced-transcripts", required=True, help="JSONL: {id,text}")
    p.add_argument("--references", required=True, help="JSONL: {id,text}")
    args=p.parse_args()
    def read(x): return {r["id"]:r["text"] for r in map(json.loads,Path(x).read_text().splitlines())}
    raw,enh,ref=read(args.raw_transcripts),read(args.enhanced_transcripts),read(args.references)
    raw_w=[]; enh_w=[]; refs=[]; raw_hyps=[]; enh_hyps=[]
    for k,r in ref.items():
        if k in raw and k in enh:
            raw_w.append(wer(r,raw[k]))
            enh_w.append(wer(r,enh[k]))
            refs.append(r); raw_hyps.append(raw[k]); enh_hyps.append(enh[k])
    if not raw_w: raise SystemExit("No matching transcript IDs")

    # Keep the original macro-average for continuity, and also report pooled
    # WER (total edits / total reference words), which is the more standard
    # corpus-level ASR metric.
    macro_raw=sum(raw_w)/len(raw_w); macro_enh=sum(enh_w)/len(enh_w)
    pooled_raw=pooled_wer(refs, raw_hyps)
    pooled_enh=pooled_wer(refs, enh_hyps)
    rel_macro=(macro_raw-macro_enh)/macro_raw*100 if macro_raw else 0
    rel_pooled=(pooled_raw-pooled_enh)/pooled_raw*100 if pooled_raw else 0
    print(json.dumps({
        "n":len(raw_w),
        "macro_raw_wer":macro_raw,
        "macro_enhanced_wer":macro_enh,
        "macro_relative_wer_improvement_pct":rel_macro,
        "pooled_raw_wer":pooled_raw,
        "pooled_enhanced_wer":pooled_enh,
        "pooled_relative_wer_improvement_pct":rel_pooled
    },indent=2))

if __name__ == "__main__": main()
