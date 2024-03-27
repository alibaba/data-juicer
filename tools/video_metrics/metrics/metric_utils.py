# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import hashlib
import os
import pathlib
import pickle
import time
import uuid
from urllib.parse import urlparse

from tools.video_metrics.util import EasyDict, format_time, make_cache_dir_path, open_url
import einops
import numpy as np
import torch
from tools.video_metrics.dataset import VideoDataset, VideoDatasetPerImage
from tools.video_metrics import distributed
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from collections.abc import Iterator


def random_seed(max_seed: int = 2**31 - 1) -> int:
    seed = torch.randint(max_seed + 1, (), device="cuda")
    if distributed.get_world_size() > 1:
        dist.broadcast(seed, src=0)
    return seed.item()


def get_infinite_data_iter(dataset: Dataset, batch_size: int, seed: int = None, **loader_kwargs) -> Iterator:
    seed = random_seed() if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    sampler = DistributedSampler(dataset, seed=seed) if distributed.get_world_size() > 1 else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, generator=generator, **loader_kwargs)

    epoch = 0
    while True:
        if distributed.get_world_size() > 1:
            sampler.set_epoch(epoch)
        for sample in loader:
            yield sample
        epoch += 1


# fmt: off
#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, lr_G=None, G_kwargs={}, dataset_kwargs={}, cond_dataset_kwargs=None, loader_kwargs=None,
                 progress=None, cache=True, gen_dataset_kwargs={}, generator_as_dataset=False,
                 normalize_weighting=True, single_sample_per_video=False, verbose=False, replace_cache=False,
                 detector_path=None):
        self.verbose                  = verbose and distributed.get_rank() == 0
        self.replace_cache            = replace_cache
        self.G                        = G
        self.lr_G                     = lr_G
        self.G_kwargs                 = EasyDict(G_kwargs)
        self.dataset_kwargs           = EasyDict(dataset_kwargs)
        self.cond_dataset_kwargs      = None if cond_dataset_kwargs is None else EasyDict(cond_dataset_kwargs)
        self.loader_kwargs            = None if loader_kwargs is None else EasyDict(loader_kwargs)
        self.progress                 = progress.sub() if progress is not None and distributed.get_rank() == 0 else ProgressMonitor(verbose=self.verbose)
        self.cache                    = cache
        self.gen_dataset_kwargs       = EasyDict(gen_dataset_kwargs)
        self.generator_as_dataset     = generator_as_dataset
        self.normalize_weighting      = normalize_weighting
        self.single_sample_per_video  = single_sample_per_video
        self.detector_path            = detector_path

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, verbose=False):
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    key = (url, torch.device('cuda'))

    if key not in _feature_detector_cache:

        if world_size > 1 and rank != 0:
            # Wait for rank 0 so that only one rank downloads detector to cache.
            torch.distributed.barrier()

        with open_url(url, verbose=(verbose and rank == 0)) as f:
            if urlparse(url).path.endswith('.pkl'):
                _feature_detector_cache[key] = pickle.load(f).requires_grad_(False)
            else:
                _feature_detector_cache[key] = torch.jit.load(f)

            _feature_detector_cache[key].eval().to("cuda")

        if world_size > 1 and rank == 0:
            # Ranks other than 0 are now free to load detector.
            torch.distributed.barrier()

    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def gather_interleave(x):
    world_size = distributed.get_world_size()
    if world_size > 1:
        ys = []
        for src in range(world_size):
            y = x.clone()
            torch.distributed.broadcast(y, src=src)
            ys.append(y)
        x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
    return x
            
class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None
        self.weight = 0.0

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x, weight=None):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        
        if weight is not None:
            assert weight.ndim == 1
            assert weight.shape[0] == x.shape[0]
                
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            
            if weight is not None:
                weight = weight[:self.max_items - self.num_items]

            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            if weight is None:
                self.raw_mean += x64.sum(axis=0)
                self.raw_cov += x64.T @ x64
                self.weight += x.shape[0]
            else:
                weight = weight.astype(np.float64)
                weighted_x64 = x64 * weight[:, None]
                self.raw_mean += weighted_x64.sum(axis=0)
                self.raw_cov += x64.T @ weighted_x64
                self.weight += weight.sum(axis=0)

    def append_torch(self, x, weight=None):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        x = gather_interleave(x).cpu().numpy()
        weight = None if weight is None else gather_interleave(weight).cpu().numpy()
        self.append(x, weight)

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.weight
        cov = self.raw_cov / self.weight
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items: int):
        assert (self.num_items is None) or (cur_items <= self.num_items), f"Wrong `items` values: cur_items={cur_items}, self.num_items={self.num_items}"
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_dataset(
    opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64,
    max_items=None, temporal_detector=False, use_image_dataset=False,
    feature_stats_cls=FeatureStats, **stats_kwargs):
    
    assert not temporal_detector or not use_image_dataset
    
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    if use_image_dataset:
        dataset = VideoDatasetPerImage(**opts.dataset_kwargs)
    else:
        dataset = VideoDataset(**opts.dataset_kwargs)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        cache_kwargs = dict(
            dataset_state=pickle.dumps(dataset),
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            stats_kwargs=stats_kwargs,
            feature_stats_cls=feature_stats_cls.__name__,
            normalize_weighting=use_image_dataset and opts.normalize_weighting,
            single_sample_per_video=not use_image_dataset and opts.single_sample_per_video,
            max_items=max_items,
        )

        cache_tag = hashlib.blake2b(repr(sorted(cache_kwargs.items())).encode(), digest_size=8).hexdigest()
        cache_file = make_cache_dir_path(
            "dj-fvd-metrics", pathlib.Path(dataset.dataset_path).parent.name,
            get_feature_detector_name(detector_url), f"{cache_tag}.pkl"
        )

        if not opts.replace_cache:
            # Check if the file exists (all processes must agree).
            flag = os.path.isfile(cache_file) if rank == 0 else False
            if world_size > 1:
                flag = torch.as_tensor(flag, dtype=torch.float32, device="cuda")
                torch.distributed.broadcast(tensor=flag, src=0)
                flag = (float(flag.cpu()) != 0)

            # Load.
            if flag:
                return feature_stats_cls.load(cache_file)

    # Initialize.
    if use_image_dataset or opts.single_sample_per_video:
        num_items = len(dataset)
        if max_items is not None:
            num_items = min(num_items, max_items)
        indices = torch.randperm(num_items)
        dataset = torch.utils.data.Subset(dataset, indices[:num_items])
    else:
        assert max_items is not None
        num_items = max_items

    stats = feature_stats_cls(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, verbose=progress.verbose)

    loader_kwargs = opts.loader_kwargs
    if loader_kwargs is None:
        loader_kwargs = dict(num_workers=2, prefetch_factor=2, pin_memory=True, persistent_workers=True)
        
    if use_image_dataset or opts.single_sample_per_video:
        item_subset = [(i * world_size + rank) % num_items for i in range((num_items - 1) // world_size + 1)]
        data_iter = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **loader_kwargs))
    else:
        data_iter = get_infinite_data_iter(dataset, batch_size, **loader_kwargs)

    # Main loop.
    while not stats.is_full():
        batch = next(data_iter)
        images = batch['video']
        weight = 1 / batch["num_samples_from_source"].cuda() if use_image_dataset and opts.normalize_weighting else None
        
        if not temporal_detector:
            if weight is not None:
                weight = einops.repeat(weight, "n -> (n t)", t=images.size(2))
            images = einops.rearrange(images, "n c t h w -> (n t) c h w")

        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])

        images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        features = detector(images.to("cuda"), **detector_kwargs)

        stats.append_torch(features, weight)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_generator(
        opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size: int=16, batch_gen=None, jit=False,
        temporal_detector=False, use_image_dataset=False, num_video_frames: int=1, feature_stats_cls=FeatureStats,
        subsample_factor: int=1, max_items=None, **stats_kwargs):

    assert not jit

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to("cuda")
    lr_G = None if opts.lr_G is None else copy.deepcopy(opts.lr_G).eval().requires_grad_(False).to("cuda")
    
    # Setup generator and load labels.
    dataset = None
    if opts.cond_dataset_kwargs is not None:
        assert not temporal_detector or not use_image_dataset
        opts.cond_dataset_kwargs.seq_length = num_video_frames + 2 * G.temporal_context

        if use_image_dataset:
            dataset = VideoDatasetPerImage(**opts.cond_dataset_kwargs)
        else:
            dataset = VideoDataset(**opts.cond_dataset_kwargs)

        if use_image_dataset or opts.single_sample_per_video:
            num_items = len(dataset)
            if max_items is not None:
                num_items = min(num_items, max_items)
            max_items = num_items
            indices = torch.randperm(num_items)
            dataset = torch.utils.data.Subset(dataset, indices[:num_items])
        else:
            assert max_items is not None
            num_items = max_items

        rank = distributed.get_rank()
        world_size = distributed.get_world_size()

        loader_kwargs = opts.loader_kwargs
        if loader_kwargs is None:
            loader_kwargs = dict(num_workers=2, prefetch_factor=2, pin_memory=True, persistent_workers=True)

        if use_image_dataset or opts.single_sample_per_video:
            item_subset = [(i * world_size + rank) % num_items for i in range((num_items - 1) // world_size + 1)]
            data_iter = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **loader_kwargs))
        else:
            data_iter = get_infinite_data_iter(dataset, batch_size, **loader_kwargs)

    # Initialize.
    stats = feature_stats_cls(max_items=max_items, **stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        weights = []
        for _i in range(batch_size // batch_gen):
            
            if dataset is None:
                use_rand_offset = True
                if lr_G is None:
                    image = G(batch_gen, seq_length=num_video_frames * subsample_factor + G.total_temporal_scale * int(use_rand_offset))
                    t0 = torch.randint(G.total_temporal_scale, (batch_gen,)) * int(use_rand_offset)
                    t1 = t0 + num_video_frames * subsample_factor
                    image = torch.stack([image[i, :, t0[i] : t1[i]] for i in range(batch_size)])
                else:
                    image = lr_G(batch_gen, seq_length=num_video_frames * subsample_factor + 2 * G.temporal_context + lr_G.total_temporal_scale * int(use_rand_offset))
                    t0 = torch.randint(lr_G.total_temporal_scale, (batch_gen,)) * int(use_rand_offset)
                    t1 = t0 + num_video_frames * subsample_factor + 2 * G.temporal_context
                    image = torch.stack([image[i, :, t0[i] : t1[i]] for i in range(batch_size)])
                    image = G(image)

            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                
                image = batch["video"].cuda()
                image = G(image)

                if use_image_dataset and opts.normalize_weighting:
                    weight = 1 / batch["num_samples_from_source"].cuda()
                    if not temporal_detector:
                        weight = einops.repeat(weight, "n -> (n t)", t=image.size(2))
                    weights.append(weight)

            image = image[:, :, ::subsample_factor]
            assert image.size(2) == num_video_frames

            image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            if not temporal_detector:
                image = einops.rearrange(image, "n c t h w -> (n t) c h w")
            images.append(image)

        images = torch.cat(images)
        images = images.repeat([1, 3, *([1] * (images.ndim - 2))]) if images.shape[1] == 1 else images
        weights = torch.cat(weights) if dataset is not None and use_image_dataset and opts.normalize_weighting else None
    
        features = detector(images, **detector_kwargs)

        stats.append_torch(features, weights)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def rewrite_opts_for_gen_dataset(opts):
    """
    Updates dataset arguments in the opts to enable the second dataset stats computation
    """
    new_opts = copy.deepcopy(opts)
    new_opts.dataset_kwargs = new_opts.gen_dataset_kwargs
    new_opts.cache = False
    return new_opts

# ----------------------------------------------------------------------------
