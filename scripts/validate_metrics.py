"""Diff our from-scratch metrics against pycocoevalcap, the implementation published numbers use.

    pip install pycocoevalcap && python -m scripts.validate_metrics

Until this passes, our numbers are internally consistent but NOT comparable to published
baselines — a hand-rolled CIDEr that differs subtly from the standard one makes every
"we beat/match baseline X" claim invalid. Run before writing any number into the README.
"""
import sys

from vidcap import metrics

CASES = [
    (["a man is riding a horse"], [["a man is riding a horse"]]),
    (["a man rides a horse on the beach"],
     [["a man is riding a horse", "a person rides a horse on sand", "a man on a beach"]]),
    (["a dog", "two people are dancing in a room"],
     [["a small dog runs", "a puppy plays"], ["people dance", "two people are dancing"]]),
    (["the the the the"], [["a man is cooking dinner"]]),
]


def main():
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        sys.exit("pycocoevalcap not installed — `pip install pycocoevalcap` (needs Java for METEOR,\n"
                 "but Bleu/Rouge/Cider are pure Python and work without it)")

    hyps = [h for hs, _ in CASES for h in hs]
    refs = [r for _, rs in CASES for r in rs]
    gts = {i: [{"caption": c} for c in r] for i, r in enumerate(refs)}
    res = {i: [{"caption": h}] for i, h in enumerate(hyps)}
    gts = {i: [c["caption"] for c in v] for i, v in gts.items()}
    res = {i: [c["caption"] for c in v] for i, v in res.items()}

    ref_bleu = Bleu(4).compute_score(gts, res)[0][3]
    ref_rouge = Rouge().compute_score(gts, res)[0]
    ref_cider = Cider().compute_score(gts, res)[0]
    ours = metrics.evaluate(hyps, refs)

    rows = [("BLEU-4", ours["BLEU-4"], ref_bleu), ("ROUGE-L", ours["ROUGE-L"], ref_rouge),
            ("CIDEr-D", ours["CIDEr-D"], ref_cider)]
    bad = False
    for name, a, b in rows:
        d = abs(a - b)
        ok = d < 1e-3 * max(1.0, abs(b))
        bad |= not ok
        print(f"{'ok ' if ok else 'DIFF'} {name:8s} ours={a:.6f}  pycocoevalcap={b:.6f}  |d|={d:.2e}")
    print("\nmetrics are comparable to published numbers." if not bad else
          "\nMISMATCH — fix vidcap/metrics.py before reporting any number.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
