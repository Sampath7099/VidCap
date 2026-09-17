"""Video Q&A accuracy on MSRVTT-QA.

  python -m scripts.evaluate_qa --ckpt qaB --limit 2000
  python -m scripts.evaluate_qa --ckpt qaB --scorer scorer --budgets 1,2,4

Exact-match accuracy, the published MSRVTT-QA metric — NOT BLEU/CIDEr, which are meaningless on
one-word answers and not comparable to any published QA number.
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch

from scripts.evaluate import load_model, make_learned, motion_indices, uniform_sel
from vidcap.config import OUT
from vidcap.data import cache_ns, load_split, uniform_indices
from vidcap.decode import greedy
from vidcap.encoder import cache_path
from vidcap.metrics import qa_accuracy, qa_accuracy_by_type


def answer(model, tok, recs, k, select, device, batch=16, max_new_tokens=6):
    """Generate one answer per record, in the order given.

    Questions are grouped by TOKEN LENGTH so every batch is padding-free. Left-padding would be
    wrong here in a way an attention mask cannot fix: GPT-2 uses absolute position embeddings and
    Qwen uses RoPE, so padding shifts each real token's position and silently changes the answer
    (measured 1.2 max logit drift). Grouping sidesteps it entirely and costs nothing.
    """
    ns = cache_ns("msrvtt_qa")
    prompts, by_len = [], {}
    for i, r in enumerate(recs):
        q = r["question"].strip()
        p = tok(q if q.endswith("?") else q + "?")["input_ids"]
        prompts.append(p)
        by_len.setdefault(len(p), []).append(i)

    preds = [None] * len(recs)
    for n, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), batch):
            grp = idxs[s:s + batch]
            frames = []
            for i in grp:
                r = recs[i]
                emb = np.load(cache_path(ns, r["video_id"]))["emb"]
                sel = select(emb, k, r) if len(emb) else uniform_indices(1, k)
                frames.append(torch.from_numpy(np.ascontiguousarray(emb[sel])).float())
            ids = torch.tensor([prompts[i] for i in grp], dtype=torch.long, device=device)
            out = greedy(model, torch.stack(frames).to(device),
                         max_new_tokens=max_new_tokens, prompt_ids=ids)
            for i, a in zip(grp, out):
                preds[i] = a.strip()
    assert all(p is not None for p in preds), "some questions were never answered"
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="msrvtt_qa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None)
    ap.add_argument("--budgets", default="4")
    ap.add_argument("--scorer", default=None, help="adds the learned selection arm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_split(args.dataset, args.split, args.root, args.limit)
    if not recs:
        raise SystemExit("no cached shards for this split — run scripts/fetch_qa.py")
    golds = [r["answer"] for r in recs]
    types = [r.get("answer_type", "") for r in recs]
    print(f"{len(recs)} {args.dataset}/{args.split} questions")

    model, meta = load_model(args.ckpt, device)
    # A QA checkpoint and a captioning checkpoint are structurally identical, so nothing would
    # error if the wrong one were passed — it would just answer badly. Say so.
    print(f"checkpoint task: {meta.get('task', 'unknown')}")

    # The floor: always answer the most common training answer. Accuracy above this is the only
    # thing that shows the model learned anything beyond the answer prior.
    prior = Counter(golds).most_common(1)[0][0]
    print(f"answer-prior floor (always {prior!r}): {qa_accuracy([prior]*len(golds), golds):.4f}")

    selectors = {"uniform": uniform_sel, "motion": motion_indices}
    if args.scorer:
        selectors["learned"] = make_learned(args.scorer, device)

    results = {"dataset": args.dataset, "split": args.split, "n": len(recs),
               "prior": prior, "curves": {}}
    for name, sel in selectors.items():
        results["curves"][name] = {}
        for k in [int(b) for b in args.budgets.split(",")]:
            preds = answer(model, model.tok, recs, k, sel, device)
            acc = qa_accuracy(preds, golds)
            by = qa_accuracy_by_type(preds, golds, types)
            results["curves"][name][k] = {"accuracy": acc,
                                          "by_type": {t: {"acc": a, "n": n} for t, (a, n) in by.items()}}
            detail = "  ".join(f"{t} {a:.3f}({n})" for t, (a, n) in by.items())
            print(f"{name:8s} K={k:<3d} acc {acc:.4f}   {detail}")
            print(f"{'':8s}        e.g. {preds[0]!r} (gold {golds[0]!r})")

    p = args.out or (OUT / f"evalqa_{args.dataset}_{args.split}.json")
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
