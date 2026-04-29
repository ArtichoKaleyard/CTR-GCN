# Project Memory

## 2026-04-29 - uv Python 3.12 CUDA 12.8 setup
- Context: CTR-GCN was deployed from the GitHub repository into a uv-managed Python 3.12 environment on WSL2.
- Decision: Use `torch==2.11.0` and `torchvision==0.26.0` from the PyTorch `cu128` wheel index, with local `torchlight` installed as an editable path dependency.
- Why: The user requires Python 3.12 and CUDA 12.8; the official PyTorch install channel exposes CUDA 12.8 wheels through `https://download.pytorch.org/whl/cu128`.
- Action/Command: `uv lock --python 3.12` and `uv sync --python 3.12`.
- Verification: `torch.__version__ == 2.11.0+cu128`, `torch.version.cuda == 12.8`, `torch.cuda.is_available() == True`, and the CTR-GCN main import path succeeds.
- Follow-up: Keep `torchpack` out of the default dependency set; its old `PaviLogger` path is optional and incompatible with Python 3.12 unless separately patched or replaced.
