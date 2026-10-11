import contextlib
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import mmh3
import torch

from flower.utils.utils import add_text, format_sftp_path

hasher = lambda x: mmh3.hash(x, signed=False)
logger = logging.getLogger(__name__)

LEGACY_MODEL_PARAMS = [
    "use_pre_vlm_selection",
    "pre_vlm_select_text_tokens",
    "pre_vlm_text_keep_mass",
    "pre_vlm_text_keep_first",
    "pre_vlm_text_min_tokens",
    "pre_vlm_text_max_tokens",
]


def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    train_folder_path = Path(train_folder)

    if train_folder_path.suffix == '.yaml':
        train_folder_path = train_folder_path.parent.parent

    train_cfg_path = None

    hf_config_path = train_folder_path / "config.yaml"
    if hf_config_path.exists():
        train_cfg_path = hf_config_path
        print(f"Detected HuggingFace format")
    elif (train_folder_path / ".hydra").exists():
        train_cfg_path = train_folder_path / ".hydra/config.yaml"
        print(f"Detected Hydra format")
    else:
        current_dir = train_folder_path
        for _ in range(3):
            if (current_dir / ".hydra").exists():
                train_cfg_path = current_dir / ".hydra/config.yaml"
                print(f"Found Hydra config in parent directory: {current_dir}")
                break
            current_dir = current_dir.parent

        if train_cfg_path is None:
            raise ValueError(f"Could not find config.yaml in either HuggingFace format ({hf_config_path}) or Hydra format ({train_folder_path}/.hydra/config.yaml)")

    train_cfg_path = format_sftp_path(train_cfg_path)
    print(f"Loading config from: {train_cfg_path}")

    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")

    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'

    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)

    if prep_dm_and_deps:
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader["lang"].dataset

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint_path = Path(checkpoint).expanduser()
    print(f"Loading model from {checkpoint_path}")

    model = load_mode_from_safetensor(
        checkpoint_path,
        overwrite_cfg=eval_cfg_overwrite.get("model", {}),
        config_path=train_cfg_path,
    )

    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def load_mode_from_safetensor(
    filepath: Path,
    overwrite_cfg: dict = {},
    config_path=None,
):
    filepath = Path(filepath)

    ckpt_path = None
    config_path = Path(config_path) if config_path is not None else None

    if filepath.is_file():
        ckpt_path = filepath

        hf_config_path = filepath.parent / "config.yaml"
        print(f"Checking for HF format: {filepath.name} == 'model.safetensors' and {hf_config_path} exists: {hf_config_path.exists()}")

        if config_path is None:
            if filepath.name == "model.safetensors" and hf_config_path.exists():
                config_path = hf_config_path
                print(f"Detected HuggingFace format")
            else:
                config_path = _find_hydra_config(filepath)
                print(f"Detected Hydra format")

    elif filepath.is_dir():
        hf_model_path = filepath / "model.safetensors"
        hf_config_path = filepath / "config.yaml"

        if hf_model_path.exists() and hf_config_path.exists():
            ckpt_path = hf_model_path
            config_path = hf_config_path
            print(f"Detected HuggingFace format")
        elif (filepath / ".hydra").exists():
            ckpt_path = filepath
            config_path = filepath / ".hydra/config.yaml"
            print(f"Detected Hydra format")
        else:
            raise ValueError(f"Directory does not contain expected format. "
                           f"Expected either: model.safetensors + config.yaml OR .hydra/config.yaml in: {str(filepath)}")
    else:
        raise ValueError(f"Path does not exist: {str(filepath)}")

    if config_path is None:
        raise ValueError(f"Could not determine config path for: {str(filepath)}")

    print(f"Loading config from: {config_path}")

    config = OmegaConf.load(config_path)

    print(f"Loading model from {ckpt_path}")

    if _is_hf_format(config_path):
        if hasattr(config, 'model'):
            load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
        else:
            load_cfg = OmegaConf.create({**OmegaConf.to_object(config), **{"optimizer": None}, **overwrite_cfg})
    else:
        load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})

    if 'ckpt_path' in load_cfg:
        del load_cfg['ckpt_path']

    selection_mode = load_cfg.pop("pre_vlm_selection_mode", "split_text_proprio")
    if selection_mode != "split_text_proprio":
        raise ValueError(f"Unsupported pre_vlm_selection_mode '{selection_mode}', expected 'split_text_proprio'")
    for param in LEGACY_MODEL_PARAMS:
        load_cfg.pop(param, None)

    load_cfg["pretrained_model_path"] = str(ckpt_path)
    load_cfg["load_pretrained"] = True

    model = hydra.utils.instantiate(load_cfg)

    print(f"Finished loading model {ckpt_path}")
    return model

def join_vis_lang(img, lang_text):
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def _find_hydra_config(filepath: Path) -> Path:
    config_dir = filepath.parent

    hydra_dir = None
    current_dir = config_dir
    for _ in range(5):
        if (current_dir / ".hydra").exists():
            hydra_dir = current_dir / ".hydra"
            break
        current_dir = current_dir.parent

    if hydra_dir is None:
        parts = filepath.parts
        try:
            idx = parts.index("best_checkpoints")
            if idx + 1 < len(parts):
                base_dir = Path(*parts[:idx+2])
                if (base_dir / ".hydra").exists():
                    hydra_dir = base_dir / ".hydra"
        except ValueError:
            pass

    if hydra_dir is None:
        raise ValueError(f"Could not find .hydra directory for checkpoint: {str(filepath)}")

    return hydra_dir / "config.yaml"


def _is_hf_format(config_path: Path) -> bool:
    return config_path.parent.name != ".hydra"
