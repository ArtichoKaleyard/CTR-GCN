"""Export CTR-GCN validation scores and compute NTU60 xsub fusion accuracy."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import pickle
import sys
from typing import NamedTuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.core.config import LoaderConfig, RuntimeConfig
from foundry.core.registry import get_dataset_spec, get_model_builder
from foundry.data import build_dataloaders
from foundry.runtime.builders import _build_model_params

from foundry_entry import get_ctrgcn_run_config, register_ctrgcn


class StreamRun(NamedTuple):
    """Description of one trained stream checkpoint."""

    name: str
    preset: str
    output_dir: Path


STREAM_RUNS = (
    StreamRun(
        name="joint",
        preset="ntu60_xsub",
        output_dir=Path("artifacts/skeleton/ntu60_ctrgcn_xsub_joint"),
    ),
    StreamRun(
        name="bone",
        preset="ntu60_xsub_bone",
        output_dir=Path("artifacts/skeleton/ntu60_ctrgcn_xsub_bone"),
    ),
    StreamRun(
        name="joint_motion",
        preset="ntu60_xsub_joint_motion",
        output_dir=Path("artifacts/skeleton/ntu60_ctrgcn_xsub_joint_motion"),
    ),
    StreamRun(
        name="bone_motion",
        preset="ntu60_xsub_bone_motion",
        output_dir=Path("artifacts/skeleton/ntu60_ctrgcn_xsub_bone_motion"),
    ),
)

DEFAULT_WEIGHTS = {
    "joint": 0.6,
    "bone": 0.6,
    "joint_motion": 0.4,
    "bone_motion": 0.4,
}


def _prepare_config(
    preset: str,
    batch_size: int,
    device: str,
    eval_p_interval: tuple[float, ...] | None,
) -> object:
    """Build an evaluation-only RunConfig for a preset."""

    config = get_ctrgcn_run_config(preset)
    if eval_p_interval is not None:
        config.dataset.params["p_interval"] = list(eval_p_interval)
    config.loader = LoaderConfig(
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    config.runtime = RuntimeConfig(
        epochs=1,
        device=device,
        seed=config.runtime.seed,
        checkpoint_interval=1,
    )
    return config


def _build_model(config: object, device: torch.device) -> torch.nn.Module:
    """Construct the registered Foundry model without creating training artifacts."""

    dataset_spec = get_dataset_spec(config.dataset.name)
    model_builder = get_model_builder(config.model.name)
    model = model_builder(_build_model_params(config, dataset_spec), device=device)
    model.eval()
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    """Load model weights and return the checkpoint epoch."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("epoch", 0))


def _export_scores(
    run: StreamRun,
    *,
    batch_size: int,
    device: torch.device,
    eval_p_interval: tuple[float, ...] | None,
    score_name: str,
) -> tuple[OrderedDict[str, np.ndarray], list[int], list[str], int]:
    """Export one stream's validation logits as a legacy score mapping."""

    config = _prepare_config(
        run.preset,
        batch_size=batch_size,
        device=str(device),
        eval_p_interval=eval_p_interval,
    )
    _, val_loader = build_dataloaders(
        config.dataset,
        config.loader,
        requested_device=str(device),
    )
    model = _build_model(config, device=device)
    checkpoint_epoch = _load_checkpoint(model, run.output_dir / "best.pt", device=device)

    score_map: OrderedDict[str, np.ndarray] = OrderedDict()
    labels: list[int] = []
    sample_names: list[str] = []
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch in val_loader:
            inputs = batch["inputs"]
            if not isinstance(inputs, dict) or len(inputs) != 1:
                raise ValueError(f"{run.name} expected a single input stream, got {inputs!r}")
            stream_tensor = next(iter(inputs.values())).to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            logits = model(stream_tensor)
            logits_np = logits.detach().cpu().numpy()

            batch_names = [str(name) for name in batch["sample_name"]]
            batch_labels = [int(label) for label in target.detach().cpu().tolist()]
            for sample_name, label, score in zip(batch_names, batch_labels, logits_np, strict=True):
                score_map[sample_name] = score.astype(np.float32, copy=False)
                labels.append(label)
                sample_names.append(sample_name)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += int(target.numel())

    score_path = run.output_dir / score_name
    with score_path.open("wb") as handle:
        pickle.dump(score_map, handle)

    accuracy = correct / total
    print(
        f"{run.name}: epoch={checkpoint_epoch} "
        f"samples={total} acc={accuracy * 100:.4f}% score={score_path}"
    )
    return score_map, labels, sample_names, checkpoint_epoch


def _compute_fusion(
    score_maps: dict[str, OrderedDict[str, np.ndarray]],
    labels: list[int],
    sample_names: list[str],
    weights: dict[str, float],
) -> tuple[float, float]:
    """Compute top-1 and top-5 accuracy from weighted stream logits."""

    top1 = 0
    top5 = 0
    for sample_name, label in zip(sample_names, labels, strict=True):
        fused_score = None
        for stream_name, score_map in score_maps.items():
            stream_score = score_map[sample_name] * weights[stream_name]
            fused_score = stream_score if fused_score is None else fused_score + stream_score
        if fused_score is None:
            raise ValueError("No stream scores were provided for fusion.")
        ranking = np.argsort(fused_score)
        top1 += int(int(np.argmax(fused_score)) == label)
        top5 += int(label in ranking[-5:])
    total = len(labels)
    return top1 / total, top5 / total


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eval-p-interval",
        type=float,
        nargs="*",
        default=[0.95],
        help="Override validation p_interval before score export. Use no values to keep config defaults.",
    )
    parser.add_argument("--score-name", default="epoch1_test_score_p095.pkl")
    args = parser.parse_args()

    register_ctrgcn()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    eval_p_interval = tuple(args.eval_p_interval) if args.eval_p_interval else None
    print(f"device={device} eval_p_interval={eval_p_interval}")

    score_maps: dict[str, OrderedDict[str, np.ndarray]] = {}
    reference_labels: list[int] | None = None
    reference_names: list[str] | None = None

    for run in STREAM_RUNS:
        score_map, labels, sample_names, _ = _export_scores(
            run,
            batch_size=args.batch_size,
            device=device,
            eval_p_interval=eval_p_interval,
            score_name=args.score_name,
        )
        if reference_labels is None:
            reference_labels = labels
            reference_names = sample_names
        elif reference_labels != labels or reference_names != sample_names:
            raise ValueError(f"{run.name} validation order does not match the reference stream.")
        score_maps[run.name] = score_map

    if reference_labels is None or reference_names is None:
        raise RuntimeError("No scores were exported.")

    top1, top5 = _compute_fusion(
        score_maps,
        labels=reference_labels,
        sample_names=reference_names,
        weights=DEFAULT_WEIGHTS,
    )
    print(
        "fusion weights="
        f"{DEFAULT_WEIGHTS} top1={top1 * 100:.4f}% top5={top5 * 100:.4f}%"
    )


if __name__ == "__main__":
    main()
