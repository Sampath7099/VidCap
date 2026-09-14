# Dataset acquisition (Phase 0)

Loaders find videos by recursive glob, so the exact nesting under each root doesn't matter —
only the root name. Point `VIDCAP_DATA` at the parent, or pass `--root` per dataset.

```
$VIDCAP_DATA/
  msrvtt/       *.mp4 + *videodatainfo*.json      (official train/val/test split read from JSON)
  tvsum/        *.mp4 + ydata-tvsum50-anno.tsv
  summe/        *.mp4|*.webm + GT/*.mat
  activitynet/  *.mp4 + train.json / val_1.json
  holdout/      your own clips — never trained on, never scored
```

| Dataset | Where | Size |
|---|---|---|
| MSR-VTT | Kaggle: search "msrvtt"; official annotations `train_val_videodatainfo.json` + `test_videodatainfo.json` | ~7 GB |
| TVSum | Kaggle mirrors of `ydata-tvsum50`; annotations are the 50-video × 20-annotator TSV | ~2.7 GB |
| SumMe | Kaggle mirrors of `SumMe`; per-video `GT/*.mat` with `gt_score` | ~2.2 GB |
| ActivityNet Captions | Captions JSON is small and official (`activity-net.org`). **Videos are the problem** — no single clean public mirror; use whatever subset a Kaggle mirror provides. The loader keeps only videos actually present. | subset |
| holdout | Self-sourced. A handful of clips, mixed short/long. | tiny |

## Notes

- **ActivityNet is the known-soft dependency.** Phase 8's stress test needs long videos with sparse
  events. If no usable video subset materializes, the substitute is TVSum/SumMe's long videos, which
  are already long-form and already have human importance labels. Decide before Phase 8, not now.
- **Holdout hygiene**: `holdout/` is a separate root and never appears in any training loader.
  `datasets.verify_no_leakage` asserts train/eval/holdout ids are disjoint and runs on every
  `build_cache` invocation.
- TVSum scores are **per 2-second shot**; SumMe scores are **per frame**. Phase 1 must align the
  1 fps candidate pool to each convention separately — do not assume they match.
