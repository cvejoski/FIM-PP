import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import optree
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils
from datasets import get_dataset_split_names
from torch import Tensor
from torch.utils.data import IterableDataset, default_collate
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.trainer_pt_utils import IterableDatasetShard

from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data_generation.sde.gp_dynamical_systems import get_dynamicals_system_data_from_yaml
from fim.models import FIMSDEConfig
from fim.utils.helper import create_class_instance, verify_str_arg

from ..data.datasets import (
    FIMDataset,
    FIMSDEDatabatch,
    FIMSDEDatabatchTuple,
    FIMSDEDataset,
    HawkesDataset,
    HeterogeneousFIMSDEDataset,
    HFDataset,
    JsonSDEDataset,
    PaddedFIMSDEDataset,
    StreamingFIMSDEDataset,
    StreamingHawkesDataset,
    TimeSeriesImputationDatasetTorch,
)
from ..trainers.utils import is_distributed
from ..utils.logging import RankLoggerAdapter
from .utils import clean_split_from_size_info, get_path_counts, sample_from_gmm


DistributedSampler = torch.utils.data.distributed.DistributedSampler


class _EpochProxySampler:
    """
    Lightweight proxy that exposes set_epoch(epoch) so Trainer can signal
    epoch boundaries to IterableDatasets (e.g., streaming datasets) in DDP.

    When set_epoch is called, it forwards the call to the underlying dataset
    if it implements set_epoch.
    """

    def __init__(self, dataset: IterableDataset):
        self._dataset = dataset

    def set_epoch(self, epoch: int):
        if hasattr(self._dataset, "set_epoch") and callable(getattr(self._dataset, "set_epoch")):
            self._dataset.set_epoch(epoch)
        # No other sampler APIs are required by Trainer


def convert_to_pandas_data_range(date: List[datetime], periods: List[int], freq: str):
    pr = [pd.date_range(d, periods=p, freq=freq) for d, p in zip(date, periods)]

    month_and_year_pairs = [[list(pair) for pair in zip(r.month.tolist(), r.year.tolist())] for r in pr]

    return month_and_year_pairs


def transform_start_field_to_time_features(batch: dict, freq: str = "1M", key: str = "target"):
    periods = list(map(len, batch[key]))
    batch["time_feat"] = convert_to_pandas_data_range(batch["start"], periods, freq)
    return batch


class BaseDataLoader:
    def __init__(self, dataset_kwargs: Optional[dict] = {}, loader_kwargs: Optional[dict] = {}):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.batch_size = loader_kwargs.pop("batch_size", 1)
        self.test_batch_size = loader_kwargs.pop("test_batch_size", 1)
        self.dataset_kwargs = dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.iter = {}
        self.dataset = {}
        self.samplers = {}

    def _init_dataloaders(self, dataset: dict[str, torch.utils.data.Dataset]):
        for n, d in dataset.items():
            clean_split_n = clean_split_from_size_info(n)
            sampler = None
            # Use DistributedSampler for map-style datasets; for IterableDataset, create
            # an epoch-proxy so Trainer can call set_epoch and the dataset can reshuffle per epoch.
            if is_distributed():
                if not isinstance(d, IterableDataset):
                    sampler = DistributedSampler(
                        d,
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=n == "train",
                    )
                else:
                    sampler = _EpochProxySampler(d)
            self.samplers[clean_split_n] = sampler
            batch_size = self.batch_size
            if clean_split_n != "train":
                batch_size = self.test_batch_size
            if batch_size == "all" and not isinstance(d, IterableDataset):
                batch_size = len(d)

            # Build DataLoader without sampler for IterableDatasets (PyTorch forbids passing sampler)
            if isinstance(d, IterableDataset):
                # If dataset yields pre-collated batches, do not let DataLoader batch again
                dataset_batches_itself = getattr(d, "yields_collated_batches", False)
                self.iter[clean_split_n] = DataLoader(
                    d,
                    drop_last=False,
                    shuffle=False,
                    batch_size=None if dataset_batches_itself else batch_size,
                    collate_fn=None if dataset_batches_itself else self._get_collate_fn(clean_split_n, d),
                    **self.loader_kwargs,
                )
            else:
                self.iter[clean_split_n] = DataLoader(
                    d,
                    drop_last=False,
                    sampler=sampler,
                    shuffle=(sampler is None) and (clean_split_n == "train"),
                    batch_size=batch_size,
                    collate_fn=self._get_collate_fn(clean_split_n, d),
                    **self.loader_kwargs,
                )

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)

    def _get_collate_fn(self, dataset_name: str, dataset: torch.utils.data.Dataset) -> Union[None, callable]:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, test_batch_size={self.test_batch_size}, dataset={self.dataset}, iter={self.iter}, samplers={self.samplers})"

    def __str__(self) -> str:
        return self.__repr__()


class FIMHFDataLoader(BaseDataLoader):
    def __init__(
        self,
        path: Union[str, Path],
        conf: Optional[str] = None,
        split: Optional[str | List[str]] = None,
        dataset_kwargs: Optional[dict] = {},
        loader_kwargs: Optional[dict] = {},
    ):
        super().__init__(dataset_kwargs, loader_kwargs)
        self.path = path
        if split is None:
            splits = get_dataset_split_names(path)
        elif isinstance(split, str):
            splits = [split]
        else:
            splits = split
        for split in splits:
            clean_split = clean_split_from_size_info(split)
            self.dataset[clean_split] = HFDataset(self.path, conf, split=split, **dataset_kwargs)

        self._init_dataloaders(self.dataset)

    def _get_collate_fn(self, dataset_name: str, dataset: Dataset) -> callable:
        return self.pad_collate_fn

    def pad_collate_fn(self, batch: List[dict]):
        max_seq_len = max([item["seq_lengths"] for item in batch]).item()
        for item in batch:
            for k, v in item.items():
                if isinstance(v, Tensor) and v.dim() != 0 and "target" not in k:
                    pad_size = max_seq_len - v.size(0)
                    if pad_size > 0:
                        item[k] = torch.cat([v, -torch.ones(pad_size, *v.shape[1:], dtype=v.dtype)], dim=0)
        return default_collate(batch)

    def __repr__(self) -> str:
        return super().__repr__()


class FIMDataLoader(BaseDataLoader):
    def __init__(self, path: dict[str, list[str | Path]], dataset_kwargs: dict, loader_kwargs: dict):
        self.max_path_count = loader_kwargs.pop("max_path_count", None)
        self.min_path_count = loader_kwargs.pop("min_path_count", 1)
        self.max_number_of_minibatch_sizes = loader_kwargs.pop("max_number_of_minibatch_sizes", None)
        self.variable_num_of_paths = loader_kwargs.pop("variable_num_of_paths", False)
        self.current_minibatch_index = 0
        super().__init__(dataset_kwargs, loader_kwargs)
        if self.variable_num_of_paths:
            assert self.max_number_of_minibatch_sizes is not None, (
                "max_number_of_minibatch_sizes must be provided if variable_num_of_paths is True"
            )
            assert self.max_path_count is not None, "max_path_conunt must be provided if variable_num_of_paths is True"

        self.path = path
        for name, paths in path.items():
            self.dataset[name] = FIMDataset(paths, **dataset_kwargs)
            if self.variable_num_of_paths and name == "train":
                self.num_paths_for_batch = get_path_counts(
                    len(self.dataset[name]),
                    self.batch_size * dist.get_world_size() if is_distributed() else self.batch_size,
                    self.max_path_count,
                    max_number_of_minibatch_sizes=self.max_number_of_minibatch_sizes,
                    min_path_count=self.min_path_count,
                )
                if loader_kwargs.get("num_workers", 0) > 0:
                    self.worker_minibatch_paths = self._distribute_path_sizes(self.num_paths_for_batch, loader_kwargs["num_workers"])

        self._init_dataloaders(self.dataset)

    def _get_collate_fn(self, dataset_name: str, dataset: torch.utils.data.Dataset) -> Union[None, callable]:
        if self.variable_num_of_paths and dataset_name == "train":
            return partial(self.var_path_collate_fn)
        return None

    def var_path_collate_fn(self, batch: List[dict]):
        num_paths = self.__fetch_path_count_for_minibatch()
        path_idxs = torch.randint(0, self.max_path_count, (num_paths,))

        def process_item(item):
            new_item = {}
            for k, v in item.items():
                if isinstance(v, Tensor) and v.dim() != 0 and v.size(0) == self.max_path_count:
                    new_item[k] = v[path_idxs]
                else:
                    new_item[k] = v
            return new_item

        batch_data = [process_item(item) for item in batch]
        return default_collate(batch_data)

    def __fetch_path_count_for_minibatch(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            worker_batches_num_paths = self.worker_minibatch_paths[worker_id]
            num_paths = worker_batches_num_paths[self.current_minibatch_index % len(worker_batches_num_paths)]
            self.current_minibatch_index += 1
        else:
            num_paths = self.num_paths_for_batch[self.current_minibatch_index]
            self.current_minibatch_index = (self.current_minibatch_index + 1) % len(self.num_paths_for_batch)
        return num_paths

    def _distribute_path_sizes(self, minibatch_sizes: List[int], num_workers: int) -> List[List[int]]:
        """Distribute minibatch sizes among workers."""
        assert num_workers > 0, "Number of workers must be greater than 0"
        minibatch_sizes_per_worker = [[] for _ in range(num_workers)]
        for i, size in enumerate(minibatch_sizes):
            worker_id = i % num_workers
            minibatch_sizes_per_worker[worker_id].append(size)
        return minibatch_sizes_per_worker


class HawkesDataLoader(BaseDataLoader):
    def __init__(self, path: dict[str, list[str | Path]], dataset_kwargs: dict, loader_kwargs: dict):
        self.max_path_count = loader_kwargs.pop("max_path_count", None)
        self.min_path_count = loader_kwargs.pop("min_path_count", 1)
        self.max_number_of_minibatch_sizes = loader_kwargs.pop("max_number_of_minibatch_sizes", None)
        self.num_inference_paths = loader_kwargs.pop("num_inference_paths", 1)
        self.num_inference_times = loader_kwargs.pop("num_inference_times", 1)
        self.variable_num_of_paths = loader_kwargs.pop("variable_num_of_paths", False)

        # Per-dataset configuration for variable_sequence_lens
        self.variable_sequence_lens = loader_kwargs.pop("variable_sequence_lens", {})
        self.full_len_ratio = loader_kwargs.pop("full_len_ratio", 0.0)

        self.min_sequence_len = loader_kwargs.pop("min_sequence_len", None)
        self.max_sequence_len = loader_kwargs.pop("max_sequence_len", None)

        self.current_minibatch_index = 0
        # Use streaming Hawkes by default; allow opt-out via dataset_kwargs.get("use_streaming", True)
        use_streaming = dataset_kwargs.pop("use_streaming", True)
        # For streaming mode, the dataset yields pre-collated batches per directory to avoid cross-directory mixing.
        # We no longer force single-worker; callers may set num_workers>0 for performance.
        if use_streaming:
            loader_kwargs = {**loader_kwargs}
            loader_kwargs.setdefault("num_workers", 0)
        super().__init__(dataset_kwargs, loader_kwargs)
        if self.variable_num_of_paths:
            assert self.max_number_of_minibatch_sizes is not None, (
                "max_number_of_minibatch_sizes must be provided if variable_num_of_paths is True"
            )
            assert self.max_path_count is not None, "max_path_conunt must be provided if variable_num_of_paths is True"

        self.path = path
        for name, paths in path.items():
            if use_streaming:
                # Streaming dataset reads items sequentially per directory ensuring uniform marks per batch
                # Pass batch_size for internal prefetch tuning; dataset still yields single items
                split_batch_size = self.batch_size if name == "train" else self.test_batch_size
                self.dataset[name] = StreamingHawkesDataset(
                    data_dirs=paths,
                    batch_size=split_batch_size,
                    **dataset_kwargs,
                )
            else:
                # In-memory dataset
                self.dataset[name] = HawkesDataset(paths, **dataset_kwargs)
            if self.variable_num_of_paths and name == "train":
                self.num_paths_for_batch = get_path_counts(
                    len(self.dataset[name]),
                    self.batch_size * dist.get_world_size() if is_distributed() else self.batch_size,
                    self.max_path_count,
                    max_number_of_minibatch_sizes=self.max_number_of_minibatch_sizes,
                    min_path_count=self.min_path_count,
                )
                if loader_kwargs.get("num_workers", 0) > 0:
                    self.worker_minibatch_paths = self._distribute_path_sizes(self.num_paths_for_batch, loader_kwargs["num_workers"])

        self._init_dataloaders(self.dataset)

    def _get_collate_fn(self, dataset_name: str, dataset: Dataset) -> Union[None, callable]:
        def custom_collate(batch):
            # Apply variable path selection only for training data when enabled
            if self.variable_num_of_paths and dataset_name == "train":
                batch = self.var_path_collate_fn(batch)

            # Apply variable sequence lengths only when enabled and bounds are valid
            if self.variable_sequence_lens.get(dataset_name, False) and torch.rand(1) > self.full_len_ratio:
                if (
                    self.min_sequence_len is not None
                    and self.max_sequence_len is not None
                    and int(self.max_sequence_len) > int(self.min_sequence_len)
                ):
                    batch = self.custom_hawkes_collate_fun(batch)
                # else: silently skip when not configured

            # Ensure seq_lengths exists for all items (needed for inference path selection)
            for item in batch:
                if "event_times" in item and "seq_lengths" not in item:
                    # Create default seq_lengths based on the actual length of event_times
                    P, L = item["event_times"].shape[:2]
                    # For each path, find the actual sequence length (assuming padded with zeros or -1)
                    seq_lengths = []
                    for p in range(P):
                        # Find first zero or negative value to determine actual length
                        times = item["event_times"][p]
                        non_zero_mask = times > 0
                        if non_zero_mask.any():
                            seq_len = non_zero_mask.sum().item()
                        else:
                            seq_len = 1  # At least one event
                        seq_lengths.append(seq_len)
                    item["seq_lengths"] = torch.tensor(seq_lengths, dtype=torch.long)

            # ALWAYS apply inference path and time selection for all datasets BEFORE collating
            batch = self.select_inference_paths(batch)
            batch = self.select_inference_times(batch)

            # Enforce uniform mark dimension per batch and set num_marks without padding
            marks_per_item = []
            for item in batch:
                if "kernel_functions" in item:
                    marks_per_item.append(int(item["kernel_functions"].shape[0]))
                elif "base_intensity_functions" in item:
                    marks_per_item.append(int(item["base_intensity_functions"].shape[0]))
                else:
                    marks_per_item.append(None)
            unique_marks = {m for m in marks_per_item if m is not None}
            if len(unique_marks) > 1:
                raise ValueError("Detected mixed mark dimensions within a single batch. Use streaming with num_workers=0 to avoid mixing.")
            inferred_marks = next(iter(unique_marks)) if len(unique_marks) == 1 else 0
            for idx, item in enumerate(batch):
                item["num_marks"] = marks_per_item[idx] if marks_per_item[idx] is not None else inferred_marks

            # Use torch default collate for the final step
            collated = default_collate(batch)

            # Collapse num_marks to a scalar so that downstream code can safely call
            # `.item()` irrespective of batch size.
            if "num_marks" in collated and isinstance(collated["num_marks"], torch.Tensor):
                if collated["num_marks"].ndim > 0:
                    # all values are identical, so just take the first element
                    collated["num_marks"] = collated["num_marks"][0]

            return collated

        return custom_collate

    def __custom_var_dim_collate_fn(self, batch):
        """
        Custom collate function to group by 'dim' and collate inner dictionaries.
        :param batch: List of dictionaries with structure {k1: v1, k2: v2, ..., '_group_dim': dim}.
        :return: A dictionary with collated inner dictionaries and '_group_dim' key.
        """
        grouped = defaultdict(list)
        for item in batch:
            grouped[item["_group_dim"]].append(item)

        for dim, inner_dict in grouped.items():
            grouped[dim] = default_collate(inner_dict)

        return grouped

    def var_path_collate_fn(self, batch: List[dict]):
        num_paths = self.__fetch_path_count_for_minibatch()
        path_idxs = torch.randint(0, self.max_path_count, (num_paths,))

        def process_item(item):
            for k, v in item.items():
                if k in ["event_times", "event_types"]:
                    item[k] = v[path_idxs]
            return item

        batch_data = [process_item(item) for item in batch]
        return batch_data

    def custom_hawkes_collate_fun(self, batch: List[dict], previous_collate_fn=None):
        """
        Collate function for Hawkes processes which samples variable sequence lengths.
        """
        if previous_collate_fn is not None:
            batch = previous_collate_fn(batch)
        upper_bound = random.randint(self.min_sequence_len + 1, self.max_sequence_len)
        lower_bound = random.randint(self.min_sequence_len, upper_bound - 1)

        def add_variable_seq_lens(item):
            item["event_times"] = item["event_times"][:, :upper_bound]
            item["event_types"] = item["event_types"][:, :upper_bound]

            P = item["event_times"].shape[0]
            seq_lens = torch.tensor(sample_from_gmm(lower_bound, upper_bound, size=P), dtype=torch.long)
            item["seq_lengths"] = seq_lens
            return item

        batch_data = [add_variable_seq_lens(item) for item in batch]

        return batch_data

    def select_inference_paths(self, batch_data: List[dict]):
        """
        Split observations into inference and context paths. Select the first 'num_inference_paths' paths for inference and remove them from the context paths.
        """
        # Early return if no items have event_times to avoid unnecessary processing
        if not any("event_times" in item for item in batch_data):
            return batch_data

        for item in batch_data:
            if "event_times" not in item:
                continue

            P = item["event_times"].shape[0]
            if P <= self.num_inference_paths:
                raise ValueError(
                    f"Number of paths {P} is less than or equal to the number of inference paths {self.num_inference_paths}. "
                    "Please increase the number of paths in the dataset or decrease the number of inference paths."
                )

            # Use pop() to remove and retrieve tensors more efficiently than del
            event_times = item.pop("event_times")
            event_types = item.pop("event_types")
            seq_lengths = item.pop("seq_lengths")

            # Split tensors using slicing (creates views, not copies)
            item["inference_event_times"] = event_times[: self.num_inference_paths]
            item["inference_event_types"] = event_types[: self.num_inference_paths]
            item["context_event_times"] = event_times[self.num_inference_paths :]
            item["context_event_types"] = event_types[self.num_inference_paths :]
            item["context_seq_lengths"] = seq_lengths[self.num_inference_paths :]
            item["inference_seq_lengths"] = seq_lengths[: self.num_inference_paths]

        return batch_data

    def select_inference_times(self, batch_data: List[dict]):
        """
        Sample inference evaluation times by distributing points across each inter-event interval
        of every inference path.  The total number of samples (num_inference_times) is split as evenly
        as possible among the intervals (seq_len - 1); any remainder is assigned one extra point
        in the first intervals.
        """
        for item in batch_data:
            if "inference_event_times" not in item:
                continue
            # Prepare output tensor for evaluation times [P_inference, num_inference_times]
            P = self.num_inference_paths
            T = self.num_inference_times
            dtype = item["inference_event_times"].dtype
            item["intensity_evaluation_times"] = torch.zeros(P, T, dtype=dtype)
            for i in range(P):
                ev_times = item["inference_event_times"][i]
                seq_len = (
                    item["inference_seq_lengths"][i].item()
                    if isinstance(item["inference_seq_lengths"], torch.Tensor)
                    else item["inference_seq_lengths"][i]
                )
                # Clamp seq_len to not exceed the actual tensor size
                seq_len = min(seq_len, ev_times.shape[0])
                # Number of intervals between successive events
                intervals = max(seq_len - 1, 0)
                if intervals <= 0:
                    # Fallback to uniform sampling up to the last event time
                    max_t = ev_times[seq_len - 1] if seq_len > 0 else 0
                    samples = torch.rand(T, dtype=dtype) * max_t
                else:
                    # Distribute samples per interval (up to remainder)
                    base = T // intervals
                    rem = T % intervals
                    counts = [base + 1 if j < rem else base for j in range(intervals)]
                    parts: List[Tensor] = []
                    for j, cnt in enumerate(counts):
                        if cnt <= 0:
                            continue
                        start = ev_times[j]
                        end = ev_times[j + 1]
                        parts.append(torch.rand(cnt, dtype=dtype) * (end - start) + start)
                    samples = torch.cat(parts) if parts else torch.empty(0, dtype=dtype)
                # Sort samples in ascending order
                item["intensity_evaluation_times"][i] = torch.sort(samples)[0]
        return batch_data

    def __fetch_path_count_for_minibatch(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            worker_batches_num_paths = self.worker_minibatch_paths[worker_id]
            num_paths = worker_batches_num_paths[self.current_minibatch_index % len(worker_batches_num_paths)]
            self.current_minibatch_index += 1
        else:
            num_paths = self.num_paths_for_batch[self.current_minibatch_index]
            self.current_minibatch_index = (self.current_minibatch_index + 1) % len(self.num_paths_for_batch)
        return num_paths

    def _distribute_path_sizes(self, minibatch_sizes: List[int], num_workers: int) -> List[List[int]]:
        """Distribute minibatch sizes among workers."""
        assert num_workers > 0, "Number of workers must be greater than 0"
        minibatch_sizes_per_worker = [[] for _ in range(num_workers)]
        for i, size in enumerate(minibatch_sizes):
            worker_id = i % num_workers
            minibatch_sizes_per_worker[worker_id].append(size)
        return minibatch_sizes_per_worker


class TimeSeriesDataLoaderTorch:
    """Datalaoder for time series data in torch format."""

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: Optional[int] = 32,
        test_batch_size: Optional[int] = 32,
        output_fields: Optional[List[str]] = None,
        loader_kwargs: Optional[dict] = {},
        dataset_name: str = "fim.data.datasets.TimeSeriesDatasetTorch",
        dataset_kwargs: Optional[dict] = {},
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_kwargs = dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.iter = {}
        self.path = path
        self.name = ds_name

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        dataset_split_names = ["train", "test", "validation"]

        self.split = verify_str_arg(split, arg="split", valid_values=dataset_split_names + [None])

        if self.split is not None:
            self.dataset = {
                self.split: create_class_instance(
                    dataset_name,
                    {
                        "path": self.path,
                        "ds_name": self.name,
                        "split": self.split,
                        "output_fields": output_fields,
                        **self.dataset_kwargs,
                    },
                )
            }
        else:
            self.dataset = {
                split_: create_class_instance(
                    dataset_name,
                    {
                        "path": self.path,
                        "ds_name": self.name,
                        "split": split_,
                        "output_fields": output_fields,
                        **self.dataset_kwargs,
                    },
                )
                for split_ in dataset_split_names
            }

        self._init_dataloaders(self.dataset)

    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None and n == "train",
                batch_size=batch_size,
                collate_fn=partial(TimeSeriesImputationDatasetTorch.collate_fn, dataset=d)
                if isinstance(d, TimeSeriesImputationDatasetTorch)
                else None,
                **self.loader_kwargs,
            )

    def __str__(self) -> str:
        dataset_desc = {k: str(v) for k, v in self.dataset.items()}
        return f"TimeSeriesDataLoaderTorch=(batch_size={self.batch_size}, test_batch_size={self.test_batch_size}, dataset={dataset_desc})"

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)


@dataclass
class SDEDataloaderConfig:
    """
    Configurations of single SDE dataloader (e.g. for train).
    Includes dataset specifications and arguments for collate function.
    """

    data_label: str
    data_type: str
    data_paths: str | list[str]
    batch_size: int
    random_grids: bool
    min_num_grid_points: int
    max_num_grid_points: int
    random_paths: bool
    min_num_paths: int
    max_num_paths: int

    @classmethod
    def from_dict(cls, label: str, config: dict):
        """
        Construct SDEDataloaderConfig for some set label (e.g. train, test, validation) from some dict, likely passed from config yaml.
        For each attribute, check if passed config is dict that contains label, e.g.
            {"train":..., "test":..., "validation":...}
        extract the value under the label and set it as attribute.
        Otherwise, extract the value from the dict directly for this attribute, e.g. attribute = config.get(attribute_label)

        Args:
            label (str): set label, e.g. train, test, validation
            config (dict): contains attributes to construct SDEDataloaderConfig with, e.g.:
                {
                "attribute_1": value_1,
                "attribute_2": {"train": value_2_train, "test": value_2_test, "validation": value_2_validation}
                }

        Returns:
            SDEDataloaderConfig with attributes extracted from config under label.
        """
        data_type: str = cls._get_config_as_dict(label, "data_type", config, expected_type=str)
        data_paths = cls._get_config_as_dict(label, "data_paths", config, expected_type=str)
        batch_size = cls._get_config_as_dict(label, "batch_size", config, expected_type=int)

        random_grids = cls._get_config_as_dict(label, "random_grids", config, expected_type=bool)
        min_num_grid_points = cls._get_config_as_dict(label, "min_num_grid_points", config, expected_type=int)
        max_num_grid_points = cls._get_config_as_dict(label, "max_num_grid_points", config, expected_type=int)

        random_paths = cls._get_config_as_dict(label, "random_paths", config, expected_type=bool)
        min_num_paths = cls._get_config_as_dict(label, "min_num_paths", config, expected_type=int)
        max_num_paths = cls._get_config_as_dict(label, "max_num_paths", config, expected_type=int)

        return cls(
            label,
            data_type,
            data_paths,
            batch_size,
            random_grids,
            min_num_grid_points,
            max_num_grid_points,
            random_paths,
            min_num_paths,
            max_num_paths,
        )

    @staticmethod
    def _get_config_as_dict(data_label: str, attribute_label: str, config: dict, expected_type: object) -> dict:
        """
        Extract value from config dict under key `attribute_label`.
        If value is dict e.g. {"train":..., "test":..., "validation":...}, extract value under the `data_label`.
        """
        attribute_value = config.get(attribute_label)

        if attribute_value is None:
            return None

        else:
            assert isinstance(attribute_value, expected_type) or isinstance(attribute_value, dict), (
                f"Got wrong type {type(attribute_value)} for data_label {data_label} and attribute {attribute_label}."
            )

            # check if extracted value is dict and if key `data_label` can be extracted.
            if isinstance(attribute_value, dict) and data_label in list(attribute_value.keys()):
                _attribute_value = attribute_value.get(data_label)
            else:
                _attribute_value = attribute_value

            return _attribute_value


class FIMSDEDataloader(BaseDataLoader):
    """
    Dataloader for FIM SDE model.
    """

    def __init__(self, **kwargs):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.num_workers: int = kwargs.get("num_workers", 2)

        self.data_in_files: dict = kwargs.get("data_in_files")
        self.dataloader_config = {
            "train": SDEDataloaderConfig.from_dict("train", kwargs),
            "validation": SDEDataloaderConfig.from_dict("validation", kwargs),
            "test": SDEDataloaderConfig.from_dict("test", kwargs),
        }

        self.dataset = {
            "train": self._get_dataset(self.dataloader_config["train"]),
            "validation": self._get_dataset(self.dataloader_config["validation"]),
            "test": self._get_dataset(self.dataloader_config["test"]),
        }

        self.samplers = {}

        self.iter = {
            "train": self._init_dataloader(self.dataset["train"], self.dataloader_config["train"]),
            "validation": self._init_dataloader(self.dataset["validation"], self.dataloader_config["validation"]),
            "test": self._init_dataloader(self.dataset["test"], self.dataloader_config["test"]),
        }

    def _get_dataset(self, config: SDEDataloaderConfig):
        """
        Build dataset for a dataloader based on SDEDataloaderConfig. Major difference is `synthetic` or `theory` data.

        Args:
            config (SDEDataloaderConfig): Config for one dataloader (e.g. train).

        Return:
            dataset (FIMSDEDataset): Dataset with data from files or generated from dynamical systems.
        """
        if config.data_type == "synthetic":
            dataset = FIMSDEDataset.from_data_paths(config.data_paths, self.data_in_files)
        elif config.data_type == "theory":
            yaml_path = config.data_paths
            systems_data: list[FIMSDEDatabatch] = get_dynamicals_system_data_from_yaml(yaml_path, config.data_label)
            dataset = FIMSDEDataset.from_data_batches(systems_data)

        return dataset

    def update_kwargs(self, kwargs: dict | FIMDatasetConfig | FIMSDEConfig):
        assert self.dataset["train"].max_dimension == self.dataset["test"].max_dimension == self.dataset["validation"].max_dimension
        assert self.dataset["train"].max_time_steps == self.dataset["test"].max_time_steps == self.dataset["validation"].max_time_steps, (
            "max_time_steps are not equal"
        )
        assert (
            self.dataset["train"].max_location_size
            == self.dataset["test"].max_location_size
            == self.dataset["validation"].max_location_size
        ), "max_location_size are not equal"
        assert self.dataset["train"].max_num_paths == self.dataset["test"].max_num_paths == self.dataset["validation"].max_num_paths, (
            "max_num_paths are not equal"
        )

        if isinstance(kwargs, dict):
            if "dataset" in kwargs.keys():
                kwargs["dataset"]["max_dimension"] = self.dataset["train"].max_dimension
                kwargs["dataset"]["max_time_steps"] = self.dataset["train"].max_time_steps
                kwargs["dataset"]["max_location_size"] = self.dataset["train"].max_location_size
                kwargs["dataset"]["max_num_paths"] = self.dataset["train"].max_num_paths

                kwargs["model"]["max_dimension"] = self.dataset["train"].max_dimension
                kwargs["model"]["max_time_steps"] = self.dataset["train"].max_time_steps
                kwargs["model"]["max_location_size"] = self.dataset["train"].max_location_size
                kwargs["model"]["max_num_paths"] = self.dataset["train"].max_num_paths
            else:
                kwargs["max_dimension"] = self.dataset["train"].max_dimension
                kwargs["max_time_steps"] = self.dataset["train"].max_time_steps
                kwargs["max_location_size"] = self.dataset["train"].max_location_size
                kwargs["max_num_paths"] = self.dataset["train"].max_num_paths
                return kwargs

        elif isinstance(kwargs, (FIMSDEConfig, FIMDatasetConfig)):
            kwargs.max_dimension = self.dataset["train"].max_dimension
            kwargs.max_time_steps = self.dataset["train"].max_time_steps
            kwargs.max_location_size = self.dataset["train"].max_location_size
            kwargs.max_num_paths = self.dataset["train"].max_num_paths

        return kwargs

    @staticmethod
    def fimsde_collate_fn(batch: List, config: SDEDataloaderConfig):
        """
        Custom collate function to adjust number of paths and grids dynamically.

        Args:
            batch (list[FIMSDEDatabatchTuple]): Returned by Dataset's __getitem__.
            config (SDEDataloaderConfig): Configuration of randomness

        Returns:
            A new FIMSDEDatabatchTuple with randomly selected number of paths and grids.
        """
        # Extract all fields from the batch (list of FIMSDEDatabatchTuple)
        obs_times = torch.stack([item.obs_times for item in batch])
        obs_values = torch.stack([item.obs_values for item in batch])
        drift_at_locations = torch.stack([item.drift_at_locations for item in batch])
        diffusion_at_locations = torch.stack([item.diffusion_at_locations for item in batch])
        locations = torch.stack([item.locations for item in batch])
        dimension_mask = torch.stack([item.dimension_mask for item in batch])

        if hasattr(batch[0], "obs_mask") and batch[0].obs_mask is not None:
            obs_mask = torch.stack([item.obs_mask for item in batch])

        else:
            obs_mask = torch.ones_like(obs_times)

        # determine and truncate number of paths
        if config.random_paths is True:
            max_paths = obs_values.shape[1]
            indices = torch.randperm(max_paths)

            number_of_paths = torch.randint(config.min_num_paths, min(config.max_num_paths, max_paths) + 1, size=(1,))
            number_of_paths = number_of_paths.item()

            indices = indices[:number_of_paths]
            obs_times = obs_times[:, indices]
            obs_values = obs_values[:, indices]
            obs_mask = obs_mask[:, indices]

        # determine and truncate number of grids
        if config.random_grids is True:
            indices = torch.randperm(locations.shape[1])

            if config.min_num_grid_points is not None and config.min_num_grid_points != -1:
                # Randomly select number of paths for the entire batch
                max_grids = locations.size(1)
                number_of_grids = torch.randint(config.min_num_grid_points, min(config.max_num_grid_points, max_grids) + 1, size=(1,))
                number_of_grids = number_of_grids.item()

                indices = indices[:number_of_grids]
                drift_at_locations = drift_at_locations[:, indices]
                diffusion_at_locations = diffusion_at_locations[:, indices]
                locations = locations[:, indices]
                dimension_mask = dimension_mask[:, indices]

        # Return a new FIMSDEDatabatchTuple
        return FIMSDEDatabatchTuple(
            obs_times=obs_times,
            obs_values=obs_values,
            obs_mask=obs_mask,
            drift_at_locations=drift_at_locations,
            diffusion_at_locations=diffusion_at_locations,
            locations=locations,
            dimension_mask=dimension_mask,
        )

    def _init_dataloader(self, dataset: FIMSDEDataset, config: SDEDataloaderConfig) -> DataLoader:
        """
        Build dataloader (e.g. for train) from (previously constructed) dataset based on some configuration.
        """
        sampler = None
        if is_distributed():
            sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        self.samplers[config.data_label] = sampler

        return DataLoader(
            dataset,
            drop_last=False,
            sampler=sampler,
            shuffle=sampler is None,
            batch_size=config.batch_size,
            collate_fn=partial(self.fimsde_collate_fn, config=config),
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class FIMSDEDataloaderIterableDataset(BaseDataLoader):
    """
    Dataloader for FIM SDE model.
    """

    def __init__(self, num_workers: Optional[int] = 2, **kwargs):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        num_workers

        if isinstance(num_workers, int):
            self.num_workers: dict = {
                "train": num_workers,
                "test": num_workers,
                "validation": num_workers,
            }
        else:  # assume num_workers has the above structure
            self.num_workers: dict = num_workers

        # kwargs contain config of each dataset
        dataset_config: dict[str, dict] = {
            "train": self._get_dataset_config("train", kwargs),
            "validation": self._get_dataset_config("validation", kwargs),
            "test": self._get_dataset_config("test", kwargs),
        }

        self.dataset: dict[str, torch.utils.data.IterableDataset] = {
            "train": self._get_dataset(**dataset_config.get("train")),
            "validation": self._get_dataset(**dataset_config.get("validation")),
            "test": self._get_dataset(**dataset_config.get("test")),
        }

        self.iter = {
            "train": self._init_dataloader(self.dataset.get("train"), self.num_workers["train"]),
            "validation": self._init_dataloader(self.dataset.get("validation"), self.num_workers["validation"]),
            "test": self._init_dataloader(self.dataset.get("test"), self.num_workers["test"]),
        }

        self.samplers = {}

    @staticmethod
    def _get_dataset_config(label: str, config: dict) -> dict:
        """
        Return dataset config for some set label (e.g. train, test, validation) from some dict, likely passed from config yaml.
        For each key, check if passed config is dict that contains label, e.g.
            {"train":..., "test":..., "validation":...}
        extract the value under the label and set it as value.
        Otherwise, extract the value from the dict directly for this key, e.g. value = config.get(key_label)

        Args:
            label (str): set label, e.g. train, test, validation
            config (dict): contains values to construct Dataset with, e.g.:
                {
                "key_1": value_1,
                "key_2": {"train": value_2_train, "test": value_2_test, "validation": value_2_validation}
                }

        Returns:
            config_for_label (dict): Key, values extracted from config under label.
        """
        config_for_label = {}

        for key, value in config.items():
            if isinstance(value, dict) and label in value.keys():
                config_for_label[key] = value[label]

            else:
                config_for_label[key] = value

        return config_for_label

    def _init_dataloader(self, dataset: torch.utils.data.IterableDataset, num_workers: int) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset,
            drop_last=False,
            batch_size=None,  # handled by iterable dataset
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            pin_memory=True,
        )

    def _get_dataset(self, **dataset_config):
        dataset_name = dataset_config.pop("dataset_name")
        shard = dataset_config.pop("shard", False)

        if dataset_name == "HeterogeneousFIMSDEDataset":
            dataset_class = HeterogeneousFIMSDEDataset
        elif dataset_name == "PaddedFIMSDEDataset":
            dataset_class = PaddedFIMSDEDataset
        elif dataset_name == "StreamingFIMSDEDataset":
            dataset_class = StreamingFIMSDEDataset
        elif dataset_name == "JsonSDEDataset":
            dataset_class = JsonSDEDataset
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

        dataset = dataset_class(**dataset_config)

        return (
            IterableDatasetShard(dataset, num_processes=dist.get_world_size(), process_index=dist.get_rank(), drop_last=True)
            if shard is True
            else dataset
        )


def fimdataset_to_sde_collate_fn(batch: List[dict]):
    """
    Collate FIMDataset dict output into FIMSDEDatabatchTuple (for model input) and extraneous dict (for evaluation).

    Args:
        batch (list[dict]): Returned by FIMDataset's __getitem__.

    Returns:
        databatch (FIMSDEDatabatchTuple): Input for FIMSDE.
        extra_data (dict): Extraneous data for evaluation.
    """
    batch: dict = optree.tree_map(lambda *x: torch.stack(x, dim=0), *batch)

    databatch: FIMSDEDatabatchTuple = FIMSDEDatabatchTuple(
        obs_values=batch.get("obs_values"),
        obs_times=batch.get("obs_times"),
        drift_at_locations=batch.get("drift_at_locations"),
        diffusion_at_locations=batch.get("diffusion_at_locations"),
        locations=batch.get("locations"),
        dimension_mask=batch.get("dimension_mask"),
    )

    for key in ["obs_values", "obs_times", "drift_at_locations", "diffusion_at_locations", "locations", "dimension_mask"]:
        batch.pop(key, None)

    return databatch, batch


class DataLoaderFactory:
    """Dataloader factory class."""

    object_types = {}

    @classmethod
    def register(cls, object_type: str, object_class: BaseDataLoader) -> None:
        """Register new dataloader type to the factory.

        Args:
            object_type (str): name of the object
            object_class (BaseDataLoader): class that is registered
        """
        cls.object_types[object_type] = object_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataLoader:
        """Create new dataloader object.

        Args:
            object_type (str): name of the object type that is created

        Raises:
            ValueError: if the object type is not registered

        Returns:
            BaseDataLoader: new instance of the dataloader object
        """
        object_class = cls.object_types.get(name)
        if object_class:
            return object_class(**kwargs)
        else:
            raise ValueError("Invalid object type!")


DataLoaderFactory.register("ts_torch_dataloader", TimeSeriesDataLoaderTorch)
DataLoaderFactory.register("FIMDataLoader", FIMDataLoader)
DataLoaderFactory.register("FIMHFDataLoader", FIMHFDataLoader)
DataLoaderFactory.register("HawkesDataLoader", HawkesDataLoader)
DataLoaderFactory.register("FIMSDEDataloader", FIMSDEDataloader)
DataLoaderFactory.register("FIMSDEDataloaderIterableDataset", FIMSDEDataloaderIterableDataset)
