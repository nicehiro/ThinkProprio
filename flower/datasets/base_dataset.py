import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import mmh3
import torch
from torch.utils.data import Dataset

from flower.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

hasher = lambda x: mmh3.hash(x, signed=False)
logger = logging.getLogger(__name__)


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class BaseDataset(Dataset):
    def __init__(
        self,
        datasets_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        num_workers: int,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        window_sampling_strategy: str = 'random',
        geometric_p_value: float = 0.1,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        assert window_sampling_strategy in ('random', 'geometric')
        self.window_sampling_strategy = window_sampling_strategy
        self.geometric_p_value = geometric_p_value
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder
        self.aux_lang_loss_window = aux_lang_loss_window
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
        sequence = self._get_sequences(idx, window_size)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}
        seq_dict["idx"] = idx
        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            if self.window_sampling_strategy == 'geometric':
                p = self.geometric_p_value
                while True:
                    sampled_window_size = 1 + np.random.geometric(p)
                    if self.min_window_size <= sampled_window_size <= max_window:
                        return sampled_window_size
            else:
                return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        if not self.relative_actions:
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info
