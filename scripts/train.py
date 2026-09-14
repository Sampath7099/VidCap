"""Stage B (connector only) / Stage C (connector + LoRA). Resumable; safe to re-run after a kill.

  python -m scripts.train --stage B --epochs 5
  python -m scripts.train --stage C --init stageB --epochs 3
  python -m scripts.train --stage B --blind --name blind   # the control
"""
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vidcap import checkpoint
from vidcap.data import VideoCaptionDataset, load_split, make_collate
from vidcap.model import VideoCaptioner


def build(args, device):
    m = VideoCaptioner(connector=args.connector, n_prefix=args.n_prefix,
                       lora_r=args.lora_r if args.stage == "C" else 0,
                       blind=args.blind).to(device)
    if args.stage == "C" and args.init:
        ck = checkpoint.load(args.init, map_location=device)
        if ck is None:
            raise SystemExit(f"--init {args.init}: no such checkpoint (run stage B first)")
        missing = m.load_state_dict(ck["model"], strict=False)
        print(f"init from {args.init} @ step {ck['step']} "
              f"({len(missing.missing_keys)} new keys = LoRA adapters)")
    return m


def evaluate_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for frames, ids, mask in loader:
            loss, _ = model(frames.to(device), ids.to(device), mask.to(device))
            tot += loss.item() * ids.size(0)
            n += ids.size(0)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["B", "C"], required=True)
    ap.add_argument("--dataset", default="msrvtt")
    ap.add_argument("--root", default=None)
    # meanpool + expanding projector = the ClipCap/LLaVA-shaped path, which trains
    # reliably at this data scale. The resampler is a Phase 9 ablation, not the default.
    ap.add_argument("--connector", default="meanpool")
    ap.add_argument("--k", type=int, default=8, help="frame budget")
    ap.add_argument("--n-prefix", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--blind", action="store_true", help="control: zero the visual prefix")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)  # LLaVA alignment-stage value
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    name = args.name or f"stage{args.stage}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_recs = load_split(args.dataset, "train", args.root, args.limit)
    val_recs = load_split(args.dataset, "val", args.root, args.limit and max(args.limit // 10, 1))
    if not train_recs:
        raise SystemExit("no cached training shards — run scripts/build_cache.py first")
    print(f"{len(train_recs)} train / {len(val_recs)} val clips (cached shards only)")

    model = build(args, device)
    collate = make_collate(model.tok)
    dl = DataLoader(VideoCaptionDataset(args.dataset, train_recs, args.k), batch_size=args.bs,
                    shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    vdl = DataLoader(VideoCaptionDataset(args.dataset, val_recs, args.k, train=False),
                     batch_size=args.bs, collate_fn=collate) if val_recs else None

    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e6:.1f}M / {n_total/1e6:.0f}M total ({100*n_train/n_total:.1f}%)")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    ck = checkpoint.load(name, model, opt, map_location=device)
    step = ck["step"] if ck else 0
    if ck:
        print(f"resumed {name} @ step {step}")

    model.train()
    # mininterval: see build_cache — committed Kaggle runs log every tqdm redraw.
    bar = tqdm(total=args.epochs * len(dl), initial=step, desc=name,
               unit="step", mininterval=30)
    for ep in range(args.epochs):
        for frames, ids, mask in dl:
            loss, _ = model(frames.to(device), ids.to(device), mask.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            bar.update(1)
            if step % 25 == 0:
                bar.set_postfix(ep=ep, loss=f"{loss.item():.4f}")
            if step % args.save_every == 0:
                checkpoint.save(name, step, model, opt, args=vars(args))
        vl = evaluate_loss(model, vdl, device) if vdl else float("nan")
        print(f"== epoch {ep} done | val loss {vl:.4f}", flush=True)
        checkpoint.save(name, step, model, opt, args=vars(args), val_loss=vl)

    bar.close()
    checkpoint.save(name, step, model, opt, args=vars(args))
    print(f"saved {name} @ step {step}")


if __name__ == "__main__":
    main()
