import concurrent.futures
import copy
import json
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset
from .sim_aloha_dataset import SimAlohaDataset

register_codecs()


# Builds a replay buffer directly from the airhockey-1m manifest+shard HDF5 layout
# (h5_manifest.json + train/valid shard_*.h5, see $dataset_dir/README.md), unlike
# SimAlohaDataset's `_convert_real_to_dp_replay` which globs per-episode
# episode_*.hdf5 files and derives actions from ee_pos. Episodes here already
# store flat (T, 4) actions and 128x128 images matching the shape_meta directly,
# so no ee_pos extraction or crop/resize is needed. Images are streamed and
# encoded one episode at a time (not accumulated into one big python list like
# the sim_aloha loader does) since airhockey-1m is ~46GB of raw uint8 frames.
def _convert_airhockey_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    split: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    obs_shape_meta = shape_meta["obs"]
    rgb_keys = [
        key for key, attr in obs_shape_meta.items() if attr.get("type", "low_dim") == "rgb"
    ]
    assert len(rgb_keys) == 1, "airhockey_dataset expects exactly one rgb obs key"
    rgb_key = rgb_keys[0]
    c, h, w = tuple(shape_meta["obs"][rgb_key]["shape"])

    root_dir = Path(dataset_dir)
    manifest = json.loads((root_dir / "h5_manifest.json").read_text())
    entries = manifest["splits"][split]
    if not entries:
        raise FileNotFoundError(
            f"No episodes found for split '{split}' in {root_dir / 'h5_manifest.json'}"
        )
    n_steps = sum(entry["length"] for entry in entries)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    img_arr = data_group.require_dataset(
        name=rgb_key,
        shape=(n_steps, h, w, c),
        chunks=(1, h, w, c),
        compressor=Jpeg2k(level=50),
        dtype=np.uint8,
    )

    def img_copy(zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int) -> bool:
        try:
            img_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = img_arr[zarr_idx]
            return True
        except Exception:
            return False

    # group episodes by shard so each shard file is opened once
    shard_to_entries: dict = dict()
    for entry in entries:
        shard_to_entries.setdefault(entry["shard"], []).append(entry)

    episode_ends = list()
    action_chunks = list()
    prev_end = 0
    write_idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for shard_rel, shard_entries in shard_to_entries.items():
            with h5py.File(root_dir / shard_rel, "r") as f:
                for entry in shard_entries:
                    episode = f["episodes"][entry["key"]]
                    imgs = episode["images"][()]  # (T, H, W, C) uint8
                    actions = episode["actions"][()]  # (T, 4) float32
                    length = entry["length"]
                    assert imgs.shape[0] == length
                    assert actions.shape[0] == length
                    action_chunks.append(actions)

                    for hdf5_idx in range(length):
                        if len(futures) >= max_inflight_tasks:
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for fut in completed:
                                if not fut.result():
                                    raise RuntimeError("Failed to encode image!")
                        futures.add(
                            executor.submit(img_copy, write_idx, imgs, hdf5_idx)
                        )
                        write_idx += 1

                    prev_end += length
                    episode_ends.append(prev_end)
        completed, futures = concurrent.futures.wait(futures)
        for fut in completed:
            if not fut.result():
                raise RuntimeError("Failed to encode image!")

    assert write_idx == n_steps

    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    actions = np.concatenate(action_chunks, axis=0).astype(np.float32)
    _ = data_group.array(
        name="action",
        data=actions,
        shape=actions.shape,
        chunks=actions.shape,
        compressor=None,
        dtype=actions.dtype,
    )

    return ReplayBuffer(root)


def load_airhockey_replay_buffer(
    dataset_dir: str, split: str, use_cache: bool, shape_meta: dict
) -> ReplayBuffer:
    replay_buffer = None
    if use_cache:
        cache_zarr_path = os.path.join(dataset_dir, split, "cache.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print(f"Acquiring lock on cache for split '{split}'.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    replay_buffer = _convert_airhockey_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                        split=split,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception:
                    if os.path.isdir(cache_zarr_path):
                        shutil.rmtree(cache_zarr_path)
                    elif os.path.exists(cache_zarr_path):
                        os.remove(cache_zarr_path)
                    raise
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_airhockey_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
            split=split,
        )
    return replay_buffer


class AirHockeyDataset(SimAlohaDataset):
    """World-model dataset for the airhockey-1m manifest+shard HDF5 layout.

    Reuses SimAlohaDataset's normalization/__getitem__/sampling logic (which is
    format-agnostic); only replay buffer construction differs, since airhockey
    episodes are indexed via h5_manifest.json across multi-episode shard files
    rather than one episode_*.hdf5 file per episode, and already store flat
    (T, 4) actions instead of ee_pos.
    """

    def __init__(self, cfg: DictConfig) -> None:
        BaseImageDataset.__init__(self)

        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = cfg.aug_mode
        if cfg.aug_mode != "none":
            raise ValueError(
                f"Invalid augmentation mode: {cfg.aug_mode} (only 'none' is supported)"
            )
        self.aug = None

        self.train_split = cfg.train_split if "train_split" in cfg else "train"
        self.val_split = cfg.val_split if "val_split" in cfg else "valid"

        self.replay_buffer = load_airhockey_replay_buffer(
            dataset_dir, self.train_split, use_cache, shape_meta
        )

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.replay_buffer = load_airhockey_replay_buffer(
            self.dataset_dir, self.val_split, self.use_cache, self.shape_meta
        )
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set


def test_airhockey_dataset() -> None:
    from omegaconf import OmegaConf

    config_path = "configurations/dataset/airhockey_dataset.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.dataset_dir = "/scratch/cj2220/data/airhockey-1m"
    cfg.horizon = 10
    cfg.val_horizon = 20
    cfg.skip_frame = 1
    cfg.skip_idx = 4
    cfg.goal_sample = "intermediate"
    cfg.resolution = 128
    dataset = AirHockeyDataset(cfg)
    print(f"train len: {len(dataset)}")
    sample = dataset[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample["obs"].items()})
    print("action", sample["action"].shape)

    val_dataset = dataset.get_validation_dataset()
    print(f"val len: {len(val_dataset)}")
    val_sample = val_dataset[0]
    print(
        {k: (v.shape if hasattr(v, "shape") else v) for k, v in val_sample["obs"].items()}
    )
    print("validation dataset success!")


if __name__ == "__main__":
    test_airhockey_dataset()
