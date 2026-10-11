import logging
from pathlib import Path
from typing import Dict, Tuple, Union
import os

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import mmh3
import torch
from torch.utils.data import Dataset, ConcatDataset
from libero.libero import benchmark, get_libero_path

from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset)
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

from flower.datasets.utils.libero_utils import get_dataset
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


class LiberoMultitaskDataset(Dataset):
    def __init__(
        self,
        benchmark_name: str,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        num_workers: int,
        action_seq_len: int,
        transforms: Dict = {},
        custom_data_path: str = None,
        batch_size: int = 32,
        min_window_size: int = 11,
        max_window_size: int = 12,
        obs_seq_len: int = 1,
        task_embedding_format: str = "clip",
        datasets_dir: str = None,
    ):
        self.task_order = 0
        self.descriptions = []
        self.datasets = []
        self.create_cfg_for_libero(task_embedding_format)
        self.benchmark_name = benchmark_name
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = self.benchmark_dict[self.benchmark_name]()
        self.num_tasks = self.benchmark_instance.get_num_tasks()
        self.task_names = self.benchmark_instance.get_task_names()
        if custom_data_path is not None:
            self.datasets_default_path = custom_data_path
        else:
            self.datasets_default_path = get_libero_path("datasets")
        self.demo_files = [os.path.join(self.datasets_default_path, self.benchmark_instance.get_task_demonstration(i)) for i in range(self.num_tasks)]
        self.benchmark = get_benchmark(self.benchmark_name)(self.task_order)
        self.observation_space = obs_space
        self.libero_observation_space = {}
        self.translate_obs_space(obs_space)

        self.proprio_state = proprio_state
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self._initialize_datasets()
        self.concat_dataset = self.create_concatenated_dataset()
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        logger.info("finished loading dataset")

    def translate_obs_space(self, obs_space):
        self.libero_observation_space = {}
        self.libero_observation_space['rgb'] = obs_space['rgb_obs']

        self.libero_observation_space['low_dim'] = obs_space['state_obs']

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({'task_embedding_format': task_embedding_format,
                               'data': {'max_word_len': 25}})

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()


    def _initialize_datasets(self):
        for i in range(self.num_tasks):
            task_i_dataset, self.shape_meta = get_dataset(
                dataset_path=os.path.join(self.datasets_default_path, self.benchmark_instance.get_task_demonstration(i)),
                obs_modality=self.libero_observation_space,
                initialize_obs_utils=(i == 0),
                seq_len=self.action_seq_len,
            )
            self.descriptions.append(self.benchmark.get_task(i).language)
            self.datasets.append(task_i_dataset)

        self.task_embs = get_task_embs(self.cfg, self.descriptions)
        self.benchmark.set_task_embs(self.task_embs)
        self.datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(self.datasets, self.task_embs)]

        self.n_demos = [data.n_demos for data in self.datasets]
        self.n_sequences = [data.total_num_sequences for data in self.datasets]

    def create_concatenated_dataset(self):
        if not self.datasets:
            raise ValueError("Datasets have not been initialized.")
        concatenated_dataset = ConcatDataset(self.datasets)
        return concatenated_dataset

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
