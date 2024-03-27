# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_main.py

import os
import time
import json
import torch
import numpy as np
from tools.video_metrics import distributed
from tools.video_metrics.util import EasyDict, format_time

from . import metric_utils
from . import video_inception_score
from . import frechet_video_distance


# fmt: off
#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

def is_power_of_two(n: int) -> bool:
    return (n & (n-1) == 0) and n != 0

#----------------------------------------------------------------------------

def calc_metric(metric, num_runs: int=1, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)
    world_size = distributed.get_world_size()

    # Calculate.
    start_time = time.time()
    all_runs_results = [_metric_dict[metric](opts) for _ in range(num_runs)]
    total_time = time.time() - start_time

    # Broadcast results.
    for results in all_runs_results:
        for key, value in list(results.items()):
            if world_size > 1:
                value = torch.as_tensor(value, dtype=torch.float64, device="cuda")
                torch.distributed.broadcast(tensor=value, src=0)
                value = float(value.cpu())
            results[key] = value

    if num_runs > 1:
        results = {f'{key}_run{i+1:02d}': value for i, results in enumerate(all_runs_results) for key, value in results.items()}
        for key, value in all_runs_results[0].items():
            all_runs_values = [r[key] for r in all_runs_results]
            results[f'{key}_mean'] = np.mean(all_runs_values)
            results[f'{key}_std'] = np.std(all_runs_values)
    else:
        results = all_runs_results[0]

    # Decorate with metadata.
    return EasyDict(
        results         = EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = format_time(total_time),
        world_size      = world_size,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

@register_metric
def fvd2048_16f(opts):
    '''
        compute FVD, sample 2048 times in dataset, 16 adjacent frames each time.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16)
    return dict(fvd2048_16f=fvd)

@register_metric
def fvd2048_128f(opts):
    '''
        compute FVD, sample 2048 times in dataset, 128 adjacent frames each time.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=128)
    return dict(fvd2048_128f=fvd)

@register_metric
def fvd2048_128f_subsample8f(opts):
    '''
        compute FVD, sample 2048 times in dataset, 16 frames each time, sample 1 frame every adjacent 8 frames.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16, subsample_factor=8)
    return dict(fvd2048_128f_subsample8f=fvd)

@register_metric
def isv2048_ucf(opts):
    '''
        compute IS, sample 2048 times in dataset, 16 frames each time, split to 10 subset to compute ISs and return the mean and std of ISs.
    '''
    mean, std = video_inception_score.compute_isv(opts, num_gen=2048, num_splits=10, backbone='c3d_ucf101')
    return dict(isv2048_ucf_mean=mean, isv2048_ucf_std=std)
