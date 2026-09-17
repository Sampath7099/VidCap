"""Download MSRVTT-QA annotations. ~40MB of JSON, no video — MSRVTT-QA is built on the same
10k clips already cached, so nothing needs re-embedding.

  python -m scripts.fetch_qa

Idempotent: existing files are skipped.
"""
import sys
import urllib.request
from pathlib import Path

from vidcap.config import DATA

BASE = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt"
FILES = ("qa_train.json", "qa_val.json", "qa_test.json", "train_ans2label.json")


def main(root=None):
    out = Path(root or DATA / "msrvtt") / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        p = out / name
        if p.exists() and p.stat().st_size > 0:
            print(f"  have {name} ({p.stat().st_size/1e6:.1f} MB)")
            continue
        tmp = p.with_suffix(".tmp")
        print(f"  fetching {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
        tmp.replace(p)                      # atomic: a kill mid-download leaves no partial file
        print(f"  got {name} ({p.stat().st_size/1e6:.1f} MB)")
    print(f"annotations in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
