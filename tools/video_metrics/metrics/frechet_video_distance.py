# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py

"""Frechet Video Distance (FVD). Matches the original tensorflow implementation from
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
up to the upsampling operation. Note that this tf.hub I3D model is different from the one released in the I3D repo.
"""

import copy
import numpy as np
import scipy.linalg
from . import metric_utils
from tools.video_metrics import distributed

# fmt: off
#----------------------------------------------------------------------------

def compute_fvd(opts, max_real: int, num_gen: int, num_frames: int, subsample_factor: int=1, use_image_dataset=False):
    # Perfectly reproduced torchscript version of the I3D model, trained on Kinetics-400, used here:
    # https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
    # Note that the weights on tf.hub (used in the script above) differ from the original released weights
    if opts.detector_path is None:
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    else:
        detector_url = opts.detector_path
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    opts = copy.deepcopy(opts)
    opts.dataset_kwargs.seq_length = num_frames
    opts.dataset_kwargs.min_spacing = subsample_factor
    opts.dataset_kwargs.max_spacing = subsample_factor

    opts.gen_dataset_kwargs.seq_length = num_frames
    opts.gen_dataset_kwargs.min_spacing = subsample_factor
    opts.gen_dataset_kwargs.max_spacing = subsample_factor

    batch_size = max(1, 64 // num_frames)

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=0,
        capture_mean_cov=True, max_items=max_real, temporal_detector=True, batch_size=batch_size, use_image_dataset=use_image_dataset).get_mean_cov()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=subsample_factor)

    mu_gen, sigma_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=1, capture_mean_cov=True,
        max_items=num_gen, temporal_detector=True, batch_size=batch_size, use_image_dataset=use_image_dataset, **gen_kwargs).get_mean_cov()

    if distributed.get_rank() != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

# ----------------------------------------------------------------------------
