"""Train SAM3 on the spleen ultrasound dataset end-to-end.

Defaults produce a real training run on the full dataset. After training,
verifies the checkpoint is mask-decoder-only and that inference can load it.

Run:
    python examples/test_sam3_train.py                    # full dataset, 10 epochs
    python examples/test_sam3_train.py --epochs 20
    python examples/test_sam3_train.py --n 32 --epochs 2  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_spleen_data  # noqa: E402

from impact_team_2.train.sam import train_sam  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "runs" / "sam3_finetune"
MAX_CKPT_MB = 500  # trainable-only checkpoint should be well under this


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="cap number of samples (default: use the full dataset)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-inference-check", action="store_true",
                        help="skip the build_predictor load-check at the end")
    args = parser.parse_args()

    print(f"[train] loading spleen data")
    images, masks = load_spleen_data()
    print(f"[train] full dataset: images={images.shape}, masks={masks.shape}")

    if args.n is not None:
        n = min(args.n, images.shape[0])
        images = images[:n]
        masks = masks[:n]
        print(f"[train] capped to {n} samples")

    print(f"[train] training for {args.epochs} epoch(s), "
          f"lr={args.lr}, val_split={args.val_split}, out={args.output_dir}")

    ckpt = train_sam(
        images, masks,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    # --- checkpoint sanity checks -----------------------------------------
    assert ckpt.exists(), f"checkpoint not found at {ckpt}"
    size_mb = ckpt.stat().st_size / 1e6
    print(f"[train] checkpoint: {ckpt} ({size_mb:.1f} MB)")
    assert size_mb < MAX_CKPT_MB, (
        f"checkpoint unexpectedly large ({size_mb:.1f} MB) — "
        f"did we regress to saving the full model?"
    )

    state = load_file(str(ckpt))
    non_decoder = [k for k in state if "mask_decoder" not in k]
    print(f"[train] saved tensors: {len(state)} "
          f"(non-mask-decoder keys: {len(non_decoder)})")
    assert not non_decoder, (
        f"checkpoint contains non-mask-decoder keys: {non_decoder[:3]}..."
    )

    # --- inference load-check ---------------------------------------------
    if not args.no_inference_check:
        print(f"[train] verifying inference can load the checkpoint")
        from impact_team_2.inference._inference_sam3 import build_predictor
        predictor = build_predictor(weights_path=ckpt)
        out = predictor(images[0], "spleen", threshold=0.01)
        print(f"[train] inference OK — detections: {out['num_detections']}")

    print(f"\n[train] DONE ({ckpt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
