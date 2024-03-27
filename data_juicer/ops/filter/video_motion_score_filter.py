import sys
from contextlib import contextmanager

import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter

OP_NAME = 'video_motion_score_filter'

with AvailabilityChecking(['opencv-python'], OP_NAME):
    import cv2


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@OPERATORS.register_module(OP_NAME)
class VideoMotionScoreFilter(Filter):
    """Filter to keep samples with video motion scores within a specific range. The
    Farneback's algorith from OpenCV is used to compute dense optical flow.
    """

    _default_kwargs = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    def __init__(self,
                 min_score: float = 0.25,
                 max_score: float = sys.float_info.max,
                 sampling_fps: PositiveInt = 2,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_score: The minimum motion score to keep samples.
        :param max_score: The maximum motion score to keep samples.
        :param sampling_fps: The samplig rate of frames_per_second to
            compute optical flow.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        self.sampling_fps = sampling_fps

        self.extra_kwargs = {
            k: kwargs.get(k, v)
            for k, v in self._default_kwargs.items()
        }

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_motion_score in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_motion_score] = np.array(
                [], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        unique_motion_scores = {}
        for video_key in loaded_video_keys:
            # skip duplicate videos
            if video_key in unique_motion_scores:
                continue

            video_motion_scores = []
            with VideoCapture(video_key) as cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                valid_fps = min(max(self.sampling_fps, 1), fps)
                frame_interval = int(fps / valid_fps)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # if cannot get the second frame, use the last one
                frame_interval = min(frame_interval, total_frames - 1)

                prev_frame = None
                frame_count = -1
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame_count += 1

                    if not ret:
                        # If the frame can't be read, it could be due to
                        # a corrupt frame or reaching the end of the video.
                        break

                    # skip middle frames
                    if frame_count % frame_interval != 0:
                        continue

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame is None:
                        prev_frame = gray_frame
                        continue

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray_frame, None, **self.extra_kwargs)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    frame_motion_score = np.mean(mag)
                    video_motion_scores.append(frame_motion_score)
                    prev_frame = gray_frame

            # may due to frame corruption
            if not video_motion_scores:
                unique_motion_scores[video_key] = -1
            else:
                unique_motion_scores[video_key] = np.mean(video_motion_scores)

        sample[Fields.stats][StatsKeys.video_motion_score] = [
            unique_motion_scores[key] for key in loaded_video_keys
        ]
        return sample

    def process(self, sample):
        video_motion_scores = sample[Fields.stats][
            StatsKeys.video_motion_score]

        keep_bools = np.array([
            self.min_score <= motion_score <= self.max_score
            for motion_score in video_motion_scores
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
