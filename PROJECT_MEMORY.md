# Project Memory

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
