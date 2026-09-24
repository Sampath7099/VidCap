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
- **TVSum and SumMe scores are BOTH per-frame** — verified 2026-09-25 against the real files, not
  assumed. `len(scores) == CAP_PROP_FRAME_COUNT` exactly (ratio 1.000) on every video checked, for
  both datasets. An earlier note here claimed TVSum was per 2-second shot; that is wrong for the
  `ydata-tvsum50-anno.tsv` release. Alignment to the candidate pool is therefore the same index
  mapping for both — `scores[frame_index]` at the pool's sampled indices.
- TVSum scores are the mean over 20 annotators and land in ~[1, 5]; SumMe's `gt_score` is already
  normalised to ~[0, 1]. Correlate per-video (Spearman), never pool raw values across datasets.

## Acquisition that actually worked (2026-09-25)

One Kaggle dataset carries everything for both: **`veerchheda/iitp-summe-tvsum`** (3.7 GB).

```bash
kaggle datasets download -d veerchheda/iitp-summe-tvsum -p data/_dl
cd data/_dl
unzip -q -j iitp-summe-tvsum.zip "video/*.mp4" -d ../tvsum/
unzip -q -j iitp-summe-tvsum.zip "data/ydata-tvsum50-anno.tsv" "data/ydata-tvsum50-info.tsv" -d ../tvsum/
unzip -q -j iitp-summe-tvsum.zip "videos/*.mp4" -d ../summe/
unzip -q    iitp-summe-tvsum.zip "GT/*"        -d ../summe/
```

Yields `tvsum/` 50 videos + anno TSV (664 MB) and `summe/` 25 videos + 25 `GT/*.mat` (1.5 GB).
Both loaders parse these unmodified — no fixes were needed, contrary to the expectation recorded
in context.md. Skip the `.webm` copies in `videos/`; they duplicate the `.mp4`s.
