"""G0 cross-dataset evaluation.

Runs a trained checkpoint on three datasets and dumps metrics + efficiency stats
into runs/g0/metrics.json. Does not write tests, does not touch training data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from irstd_g0.data import build_dataset, DatasetSpec
from irstd_g0.model import build_g0_model, count_params, count_flops
from irstd_g0.metrics import MetricAccumulator, to_jsonable


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("g0_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}


def _eval_one(model, loader, device, thr) -> dict:
    model.eval()
    acc = MetricAccumulator()
    t0 = time.time()
    n_imgs = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            msk = batch["mask"]
            # Model needs H,W divisible by 32 (window 8 at 1/1, 1/2, 1/4 scales).
            # Pad bottom/right, crop prediction back -- metrics stay on original size.
            _, _, H, W = img.shape
            ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
            if ph or pw:
                img = torch.nn.functional.pad(img, (0, pw, 0, ph))
            prob = torch.sigmoid(model(img))[:, :, :H, :W].cpu()
            acc.update_batch(prob, msk)
            n_imgs += img.shape[0]
    elapsed = time.time() - t0
    summary = acc.summary()
    summary["latency_ms_per_image"] = (elapsed / max(1, n_imgs)) * 1000.0
    summary["throughput_imgs_per_s"] = n_imgs / max(1e-6, elapsed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g0_irstd1k.py")
    parser.add_argument("--checkpoint", default="runs/g0/g0_last.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+",
                        default=["irstd1k", "sirst_uavb", "sirst4"],
                        help="Subset of {irstd1k, sirst_uavb, sirst4}.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_g0_model(**cfg["model"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    thr = cfg["eval_threshold"]

    # Efficiency on a single canonical input (512x512) -- reports at H=W=512.
    try:
        flops = count_flops(model, input_size=(1, 1, 512, 512))
    except ImportError as e:
        flops = None
        print(f"[G0] WARN: {e}")
    params = count_params(model)

    out = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "params": int(params),
        "flops_512": flops,
        "datasets": {},
    }

    for name in args.datasets:
        if name == "irstd1k":
            root = cfg["data_root"]; split = cfg["test_split"]
        elif name == "sirst_uavb":
            root = cfg["sirst_uavb_root"]; split = "test"
        elif name == "sirst4":
            root = cfg["sirst4_root"]; split = "test"
        else:
            print(f"[G0] unknown dataset: {name}, skipping")
            continue
        ds = build_dataset(DatasetSpec(name=name, root=root, split=split, augment=False))
        if args.limit:
            ds.ids = ds.ids[: args.limit]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"[G0] eval {name}: {len(ds)} images")
        out["datasets"][name] = _eval_one(model, loader, args.device, thr)

    out_path = Path(cfg["run_dir"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)
    print(f"[G0] wrote metrics -> {out_path}")


if __name__ == "__main__":
    main()
