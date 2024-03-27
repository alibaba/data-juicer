# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/universome/stylegan-v/blob/master/src/metrics/video_inception_score.py

"""Inception Score (IS) from the paper "Improved techniques for training
GANs". Matches the original implementation by Salimans et al. at
https://github.com/openai/improved-gan/blob/master/inception_score/model.py
"""

import copy
import numpy as np
from . import metric_utils
from tools.video_metrics import distributed

# fmt: off
#----------------------------------------------------------------------------

def compute_isv(opts, num_gen: int, num_splits: int, backbone: str, use_image_dataset=False):
    if backbone == 'c3d_ucf101':
        # Perfectly reproduced torchscript version of the original chainer checkpoint:
        # https://github.com/pfnet-research/tgan2/blob/f892bc432da315d4f6b6ae9448f69d046ef6fe01/tgan2/models/c3d/c3d_ucf101.py
        # It is a UCF-101-finetuned C3D model.
        if opts.detector_path is None:
            detector_url = 'https://www.dropbox.com/s/jxpu7avzdc9n97q/c3d_ucf101.pt?dl=1'
        else:
            detector_url = opts.detector_path
    else:
        raise NotImplementedError(f'Backbone {backbone} is not supported.')

    num_frames = 16
    batch_size = 64

    if opts.generator_as_dataset:
        opts = copy.deepcopy(opts)
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_opts.dataset_kwargs.seq_length = num_frames
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=1)

    gen_probs = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs={}, batch_size=batch_size,
        capture_all=True, max_items=num_gen, temporal_detector=True, use_image_dataset=use_image_dataset, **gen_kwargs).get_all() # [num_gen, num_classes]

    if distributed.get_rank() != 0:
        return float('nan'), float('nan')

    scores = []
    np.random.RandomState(42).shuffle(gen_probs)
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

# ----------------------------------------------------------------------------
