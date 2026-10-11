import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
    ):
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def load_demo_info(self, filter_by_attribute=None, demos=None):
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        self._index_to_demo_id = dict()
        self._demo_id_to_start_indices = dict()
        self._demo_id_to_demo_length = dict()

        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()] for k in obs_keys}
            if load_next_obs:
                all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()] for k in obs_keys}
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def normalize_obs(self):
        def _compute_traj_stats(traj_obs_dict):
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True)
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True)
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("SequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"].astype(np.float32)
            obs_normalization_stats[k]["std"] = (np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3).astype(np.float32)
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length
        )

        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index- self.seq_length,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="obs",
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)

        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index]

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def get_dataset_sampler(self):
        return None

    def create_subdatasets(self, split_ratio):
        mid_index = len(self.demos) * split_ratio
        demos_first_half = self.demos[:mid_index]
        demos_second_half = self.demos[mid_index:]

        subdataset1 = SequenceDataset(
            hdf5_path=self.hdf5_path,
            obs_keys=self.obs_keys,
            dataset_keys=self.dataset_keys,
            frame_stack=self.n_frame_stack,
            seq_length=self.seq_length,
            pad_frame_stack=self.pad_frame_stack,
            pad_seq_length=self.pad_seq_length,
            get_pad_mask=self.get_pad_mask,
            goal_mode=self.goal_mode,
            hdf5_cache_mode=self.hdf5_cache_mode,
            hdf5_use_swmr=self.hdf5_use_swmr,
            hdf5_normalize_obs=self.hdf5_normalize_obs,
            filter_by_attribute=self.filter_by_attribute,
            load_next_obs=self.load_next_obs
        )
        subdataset1.load_demo_info(demos=demos_first_half)

        subdataset2 = SequenceDataset(
            hdf5_path=self.hdf5_path,
            obs_keys=self.obs_keys,
            dataset_keys=self.dataset_keys,
            frame_stack=self.n_frame_stack,
            seq_length=self.seq_length,
            pad_frame_stack=self.pad_frame_stack,
            pad_seq_length=self.pad_seq_length,
            get_pad_mask=self.get_pad_mask,
            goal_mode=self.goal_mode,
            hdf5_cache_mode=self.hdf5_cache_mode,
            hdf5_use_swmr=self.hdf5_use_swmr,
            hdf5_normalize_obs=self.hdf5_normalize_obs,
            filter_by_attribute=self.filter_by_attribute,
            load_next_obs=self.load_next_obs
        )
        subdataset2.load_demo_info(demos=demos_second_half)

        return subdataset1, subdataset2
