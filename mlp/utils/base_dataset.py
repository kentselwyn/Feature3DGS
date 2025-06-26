import collections
import torch
from torch.utils.data import DataLoader,get_worker_info
import logging
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format, np_str_obj_array_pattern,)
from abc import ABCMeta, abstractmethod
import omegaconf
from .tools import set_num_threads, set_seed


string_classes = (str, bytes)
logger = logging.getLogger(__name__)


def collate(batch):
    if not isinstance(batch, list):
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        numel = sum([x.numel() for x in batch])
        try:
            storage = elem.untyped_storage()._new_shared(numel)
        except AttributeError:
            storage = elem.storage()._new_shared(numel)
        return torch.stack(batch, dim=0)
    elif (elem_type.__module__ == "numpy" and elem_type.__name__ not in ("str_", "string_")):
        if elem_type.__name__ in ("ndarray", "memmap"):
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif elem is None:
        return elem
    else:
        return torch.stack(batch, 0)


def worker_init_fn(i):
    info = get_worker_info()
    if hasattr(info.dataset, "conf"):
        conf = info.dataset.conf
        set_seed(info.id + conf.seed)
        set_num_threads(conf.num_threads)
    else:
        set_num_threads(1)


class BaseDataset(metaclass=ABCMeta):
    base_default_conf = {
        "name": "???",
        "num_workers": 0,
        "train_batch_size": "???",
        "val_batch_size": "???",
        "test_batch_size": "???",
        "shuffle_training": True,
        "batch_size": 1,
        "num_threads": 1,
        "seed": 0,
        "prefetch_factor": 2,
    }
    default_conf = {}
    def __init__(self, conf):
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        logger.info(f"Creating dataset {self.__class__.__name__}")

        self._init(self.conf)
    @abstractmethod
    def _init(self, conf):
        raise NotImplementedError
    @abstractmethod
    def get_dataset(self, split):
        raise NotImplementedError
    def get_data_loader(self, split, shuffle=None, pinned=False, distributed=False):
        assert split in ["train", "val", "test"]
        dataset = self.get_dataset(split)
        try:
            batch_size = self.conf[split + "_batch_size"]
        except omegaconf.MissingMandatoryValue:
            batch_size = self.conf.batch_size
        num_workers = self.conf.get("num_workers")
        drop_last = True if split == "train" else False
        
        if num_workers==0 or num_workers is None:
            prefetch_factor = None
        else:
            prefetch_factor = self.conf.prefetch_factor

        if distributed:
            shuffle = False
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, drop_last=drop_last)
        else:
            sampler = None
            if shuffle is None:
                shuffle = split == "train" and self.conf.shuffle_training
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=pinned,
            collate_fn=collate,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
        )
