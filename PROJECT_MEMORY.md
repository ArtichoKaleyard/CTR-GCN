# Project Memory

## 2026-05-02 - Foundry skeleton preprocessing and motion streams
- Context: CTR-GCN NTU60 reproduction needed Foundry data input closer to the original feeder path and separate `joint_motion` / `bone_motion` streams.
- Decision: Pin Foundry to v0.3.9 and use dataset-level preprocessing params (`window_size`, `p_interval`, `random_rot`) plus first-class motion stream names.
- Why: Preprocessing belongs in the skeleton dataset layer before modality construction; training callbacks are too late for per-sample valid-frame crop/resize.
- Action/Command: Ran `uv lock --upgrade-package foundry` and `uv sync`, then added NTU60 xsub motion configs and preprocessing params.
- Verification: CPU single-worker DataLoader smoke checks produced `(B, 3, 64, 25, 2)` for `joint`, `bone`, `joint_motion`, `bone_motion`, `no_ctr`, and `no_dynamic`; CUDA single-batch forward/backward passed with `batch_size=16`.
- Follow-up: Use the same Foundry v0.3.9 params when adding xview/NTU120 motion configs. On current WSL2, keep these long-running configs at `num_workers: 0` / `persistent_workers: false`; CUDA 4-batch pressure tests showed `num_workers: 0` at about 1.5 GB RSS, `num_workers: 2` spiking to about 28.7 GB RSS, and `num_workers: 4` getting killed with exit 137.

## 2026-05-03 - NTU60 xsub four-stream fusion result
- Context: After all NTU60 xsub CTR-GCN streams finished, per-sample validation logits were needed for four-stream fusion.
- Decision: Export scores from Foundry best checkpoints with `scripts/export_ctrgcn_scores.py`, writing legacy `epoch1_test_score.pkl` files beside each stream artifact.
- Why: Foundry summaries report per-stream accuracy but do not provide the score pickle expected by the old `ensemble.py` workflow.
- Action/Command: `uv run python scripts/export_ctrgcn_scores.py --batch-size 64 --device cuda`.
- Verification: Exported 16,487 validation samples for joint, bone, joint_motion, and bone_motion. With config-default eval crop (`p=1.0`), weighted fusion `{joint: 0.6, bone: 0.6, joint_motion: 0.4, bone_motion: 0.4}` produced top1 90.0103% and top5 98.2896%. Re-exporting with official-style eval crop `p_interval=[0.95]` produced top1 90.2469% and top5 98.3805%, with scores saved as `epoch1_test_score_p095.pkl`.
- Follow-up: If reporting against paper numbers, note that this is the Foundry preprocessing/checkpoint reproduction result, not the original training stack.

## 2026-04-30 - Scope Foundry modernization around active models
- Context: Review clarified that the goal is not to preserve the old custom training stack, but to modernize active models for Foundry and PyTorch 2.11.
- Decision: Keep and modernize `model/baseline.py` because it is needed for ablation; remove the legacy `main.py` path instead of carrying a half-maintained CLI.
- Why: Active scientific assets should be first-class Foundry models, while unused old infrastructure should be either archived unchanged or removed from the modernized branch.
- Action/Command: Added baseline Foundry registration through `get_baseline_run_config()` and removed README legacy `main.py` examples.
- Verification: Ran 1-epoch CPU Foundry smoke tests for both `ctrgcn` and `baseline` with fake skeleton tensors.
- Follow-up: Treat remaining old helpers such as `feeders/`, `graph/`, and `ensemble.py` as archived/reference unless they are explicitly reintroduced into the Foundry workflow.

## 2026-04-30 - Migrated to Foundry framework
- Context: CTR-GCN training pipeline modernized from custom `main.py` + `torchlight` to Foundry v0.3.7.
- Decision: CTR-GCN and the ablation baseline stay as pure `nn.Module` implementations; Foundry entry point `foundry_entry.py` provides model builders plus pre-built experiment presets (NTU60 xsub/xview, NTU120 xsub/xset, NW-UCLA).
- Why: Foundry provides a unified runner, config system, checkpointing, and logging. No need to maintain custom training loop.
- Changes:
  - `model/ctrgcn.py`: Added `adjacency` parameter, removed deprecated `Variable`/`pdb`, cleaned up dead code.
  - `model/baseline.py`: Removed deprecated `Variable` and device-local `.cuda()` handling; kept identity-initialized adjacency for baseline ablation semantics.
  - `foundry_entry.py`: Register CTR-GCN and baseline model builders, 5 pre-built experiment configs, plus `get_baseline_run_config()` for ablations.
  - Removed: `config/`, `torchlight/`, `requirements.txt`, `conf/skeleton/`, legacy `main.py`.
  - Added Foundry[logging] as git dependency (includes herald v0.5.1).
- Known issues: Foundry #13 (NW-UCLA one-hot label off-by-one) — fixed upstream in Foundry 0.3.7 (commit 0c6d63c). Upgrade to latest Foundry main to pick up the fix.
- NW-UCLA `label_base` parameter now available in Foundry's dataset builder to disambiguate 0/1-based labels.

## 2026-04-29 - uv Python 3.12 CUDA 12.8 setup
- Context: CTR-GCN was deployed from the GitHub repository into a uv-managed Python 3.12 environment on WSL2.
- Decision: Use `torch==2.11.0` and `torchvision==0.26.0` from the PyTorch `cu128` wheel index.
- Why: The user requires Python 3.12 and CUDA 12.8; the official PyTorch install channel exposes CUDA 12.8 wheels through `https://download.pytorch.org/whl/cu128`.
- Action/Command: `uv lock --python 3.12` and `uv sync --python 3.12`.
- Verification: `torch.__version__ == 2.11.0+cu128`, `torch.version.cuda == 12.8`, `torch.cuda.is_available() == True`.
- Follow-up: `torchlight` replaced by Foundry; `torchpack`/PaviLogger no longer relevant.
