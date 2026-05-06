# CTR-GCN

This repo is the official implementation for [Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213) (ICCV 2021).

Note: We also provide a simple and strong baseline model, which achieves 83.7% on NTU120 CSub with joint modality only.

## Architecture of CTR-GC
![image](src/framework.jpg)

# Prerequisites

- Python >= 3.12
- PyTorch == 2.11.0 (CUDA 12.8)
- Dependencies managed via [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

# Data Preparation

### Download datasets

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   - `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
3. Extract above files to `./data/nturgbd_raw/`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

Generate preprocessed `.npz` files:

```bash
cd ./data/ntu   # or cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

# Training

## Via Foundry (recommended)

```python
from foundry_entry import register_ctrgcn, get_ctrgcn_run_config
from foundry.runtime.runner import train_once

register_ctrgcn()
run_config = get_ctrgcn_run_config("ntu60_xsub")
train_once(run_config)
```

Available presets: `ntu60_xsub`, `ntu60_xsub_bone`, `ntu60_xsub_joint_motion`,
`ntu60_xsub_bone_motion`, `ntu60_xview`, `ntu120_xsub`, `ntu120_xset`, `nw_ucla`.
Topology ablation presets: `ntu60_xsub_no_ctr`, `ntu60_xsub_no_dynamic`,
`ntu60_xsub_q_only`.

Baseline ablation uses the same dataset presets:

```python
from foundry_entry import register_ctrgcn, get_baseline_run_config
from foundry.runtime.runner import train_once

register_ctrgcn()
run_config = get_baseline_run_config("ntu120_xsub")
train_once(run_config)
```

Configurable runs:

```python
run_config = get_ctrgcn_run_config("nw_ucla")
run_config.optimizer.learning_rate = 0.05
train_once(run_config)
```

# Testing

Use Foundry checkpoints and summaries under the configured `artifacts/` output directory.

# Ensemble

```bash
# Ensemble four modalities of CTRGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub \
  --joint-dir work_dir/ntu120/csub/ctrgcn \
  --bone-dir work_dir/ntu120/csub/ctrgcn_bone \
  --joint-motion-dir work_dir/ntu120/csub/ctrgcn_motion \
  --bone-motion-dir work_dir/ntu120/csub/ctrgcn_bone_motion
```

# Pretrained Models

Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing). Put files to `<work_dir>` and run **Testing** command.

# Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

# Citation

Please cite this work if you find it useful:

      @inproceedings{chen2021channel,
        title={Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition},
        author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={13359--13368},
        year={2021}
      }

# Contact
For any questions, feel free to contact: `chenyuxin2019@ia.ac.cn`
