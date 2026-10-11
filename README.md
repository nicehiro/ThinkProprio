<div align="center">

# Think Proprioceptively: State-Grounded Visual Token Selection for VLA Policies

[![arXiv](https://img.shields.io/badge/arXiv-2602.06575-b31b1b.svg)](https://arxiv.org/abs/2602.06575)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nicehiro.github.io/ThinkProprio/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/fywang/ThinkProprio-CALVIN-ABC-D)

</div>

ThinkProprio discretizes proprioception into VLM-vocabulary tokens and uses them together with the language instruction to select visual patches before the VLM. An instruction branch and a proprioception branch each vote for visual tokens; the union of both selections, plus a global context token, is passed to the VLM. The model is built on [FLOWER](https://github.com/intuitive-robots/flower_vla_calvin).

## Installation

Requires Linux, CUDA 12, and Python 3.9.

```bash
git clone --recurse-submodules https://github.com/nicehiro/ThinkProprio.git
cd ThinkProprio
uv sync --frozen
```

## Data

### CALVIN

```bash
wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
unzip task_ABC_D.zip
uv run --no-sync preprocess/extract_by_key.py -i /path/to/calvin --in_task task_ABC_D
```


### LIBERO

```bash
uv run LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

## CALVIN ABC→D

### Training

```bash
WANDB_MODE=offline uv run --no-sync flower/training_calvin.py \
    root_data_dir=/path/to/calvin/task_ABC_D
```

### Evaluation

[Download the CALVIN ABC→D checkpoint](https://huggingface.co/fywang/ThinkProprio-CALVIN-ABC-D/tree/v1.0). The reference EMA model achieves **4.515** over 1,000 chains with a 1,000-step limit per subtask.

```bash
CALVIN_MODEL_DIR="$(pwd)/checkpoints/calvin_abc_d"
uv run --no-sync hf download fywang/ThinkProprio-CALVIN-ABC-D \
    --revision v1.0 --local-dir "$CALVIN_MODEL_DIR"

uv run --no-sync flower/evaluation/flower_evaluate.py \
    train_folder="$CALVIN_MODEL_DIR" \
    checkpoint="$CALVIN_MODEL_DIR/model.safetensors" \
    dataset_path=/path/to/calvin/task_ABC_D
```

## LIBERO

### Training

```bash
uv run flower/training_libero.py libero_benchmark=libero_spatial
```


## Acknowledgements

This code builds on [FLOWER](https://github.com/intuitive-robots/flower_vla_calvin), [CALVIN](https://github.com/mees/calvin), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [LightVLA](https://github.com/LiAutoAD/LightVLA).

## Citation

```bibtex
@inproceedings{wang2026thinkproprio,
  title={Think Proprioceptively: State-Grounded Visual Token Selection for VLA Policies},
  author={Wang, Fangyuan and Zhou, Peng and Qi, Jiaming and Lyu, Shipeng and He, Chengyang and Navarro-Alarcon, David and Guo, Guodong},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
