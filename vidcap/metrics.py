"""Captioning metrics from scratch: BLEU-4, ROUGE-L, CIDEr-D.

Hand-rolled for the from-scratch ledger, but published-number comparability matters more than
purity — `scripts/validate_metrics.py` diffs these against pycocoevalcap. Do not report numbers
against published baselines until that validation has been run.
"""
import math
import re
from collections import Counter, defaultdict

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def tokenize(s):
    return _PUNCT.sub(" ", s.lower()).split()


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


# --- BLEU-4 (corpus-level, as reported in captioning papers) --------------------

def bleu(hyps, refs, max_n=4):
    """hyps: [str]; refs: [[str]]. Corpus-level BLEU-n with brevity penalty."""
    clipped, total = [0] * max_n, [0] * max_n
    hyp_len, ref_len = 0, 0
    for h, rs in zip(hyps, refs):
        ht, rts = tokenize(h), [tokenize(r) for r in rs]
        hyp_len += len(ht)
        # brevity penalty uses the reference length closest to the hypothesis
        ref_len += min((abs(len(r) - len(ht)), len(r)) for r in rts)[1] if rts else 0
        for n in range(1, max_n + 1):
            hc = _ngrams(ht, n)
            if not hc:
                continue
            mx = Counter()
            for rt in rts:
                rc = _ngrams(rt, n)
                for g, c in rc.items():
                    mx[g] = max(mx[g], c)
            clipped[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            total[n - 1] += sum(hc.values())

    if min(total) == 0 or min(clipped) == 0:
        return 0.0
    logp = sum(math.log(clipped[i] / total[i]) for i in range(max_n)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(logp)


# --- ROUGE-L (LCS F-measure, max over references) -------------------------------

def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyps, refs, beta=1.2):
    """Max precision and max recall are taken independently across references (possibly from
    different ones) and only then combined — this is what pycocoevalcap does, and taking the
    best per-reference F-score instead gives subtly lower, non-comparable numbers."""
    scores = []
    for h, rs in zip(hyps, refs):
        ht = tokenize(h)
        p_max = r_max = 0.0
        for r in rs:
            rt = tokenize(r)
            if not ht or not rt:
                continue
            l = _lcs(ht, rt)
            p_max = max(p_max, l / len(ht))
            r_max = max(r_max, l / len(rt))
        scores.append((1 + beta ** 2) * p_max * r_max / (r_max + beta ** 2 * p_max)
                      if p_max and r_max else 0.0)
    return sum(scores) / max(len(scores), 1)


# --- CIDEr-D --------------------------------------------------------------------

def _doc_freq(refs, max_n):
    df = defaultdict(float)
    for rs in refs:
        seen = set()
        for r in rs:
            rt = tokenize(r)
            for n in range(1, max_n + 1):
                seen.update(_ngrams(rt, n).keys())
        for g in seen:
            df[g] += 1
    return df


def _tfidf(toks, n, df, log_n_docs):
    c = _ngrams(toks, n)
    vec, norm = {}, 0.0
    for g, cnt in c.items():
        idf = log_n_docs - math.log(max(df.get(g, 0.0), 1.0))
        v = cnt * idf
        vec[g] = v
        norm += v * v
    return vec, math.sqrt(norm), len(toks)


def cider_d(hyps, refs, max_n=4, sigma=6.0):
    """tf-idf n-gram cosine with count clipping and a gaussian length penalty."""
    df = _doc_freq(refs, max_n)
    log_n = math.log(max(len(refs), 1))
    total = 0.0
    for h, rs in zip(hyps, refs):
        ht, rts = tokenize(h), [tokenize(r) for r in rs]
        per_n = []
        for n in range(1, max_n + 1):
            hv, hn, hl = _tfidf(ht, n, df, log_n)
            hc = _ngrams(ht, n)
            s = 0.0
            for rt in rts:
                rv, rn, rl = _tfidf(rt, n, df, log_n)
                rc = _ngrams(rt, n)
                # clip candidate counts to the reference's, so repetition can't inflate the score
                dot = sum(min(hc[g], rc[g]) * (hv[g] / max(hc[g], 1)) * rv[g] for g in hv if g in rv)
                if hn > 0 and rn > 0:
                    dot /= hn * rn
                s += dot * math.exp(-((hl - rl) ** 2) / (2 * sigma ** 2))
            per_n.append(s / max(len(rts), 1))
        total += 10.0 * sum(per_n) / max_n
    return total / max(len(hyps), 1)


def evaluate(hyps, refs):
    return {"BLEU-4": bleu(hyps, refs), "ROUGE-L": rouge_l(hyps, refs), "CIDEr-D": cider_d(hyps, refs)}
