import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(checkpoint_path, config_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    state = checkpoint["state_dict"]
    ema = checkpoint["callbacks"]["EMA"]["ema_weights"]
    if len(state) != len(ema):
        raise ValueError("EMA tensor count does not match the checkpoint state dict")
    for (name, weight), ema_weight in zip(state.items(), ema):
        if weight.shape != ema_weight.shape:
            raise ValueError(f"EMA shape mismatch for {name}")
    state = dict(zip(state, ema))

    training = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    model = training["model"]
    release_model = OmegaConf.load(Path(__file__).resolve().parents[1] / "conf/model/flower.yaml")
    if model["vlm_path"] != release_model.vlm_path:
        raise ValueError("The checkpoint must use the release Florence-2 model")
    model["vlm_revision"] = release_model.vlm_revision
    model["load_pretrained"] = False
    model["pretrained_model_path"] = None
    datamodule = training["datamodule"]
    datamodule["root_data_dir"] = "???"
    config = OmegaConf.create({"model": model, "datamodule": datamodule})
    OmegaConf.save(config, output_dir / "config.yaml")

    model_path = output_dir / "model.safetensors"
    print(f"Saving {len(state)} EMA tensors to {model_path}", flush=True)
    save_file(state, str(model_path), metadata={"weights": "EMA"})
    print("Computing checkpoint and bundle checksums", flush=True)
    manifest = {
        "source_checkpoint": checkpoint_path.name,
        "source_sha256": sha256(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "training_seed": training["seed"],
        "weights": "EMA",
        "tensor_count": len(state),
        "vlm_revision": str(release_model.vlm_revision),
        "files": {
            name: sha256(output_dir / name)
            for name in ("model.safetensors", "config.yaml")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.config, args.output)
