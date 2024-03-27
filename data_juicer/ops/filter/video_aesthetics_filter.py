import numpy as np
from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from loguru import logger

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

OP_NAME = 'video_aesthetics_filter'
CHECK_PKGS = ['torch', 'transformers', 'simple-aesthetics-predictor']

with AvailabilityChecking(CHECK_PKGS, OP_NAME):

    import aesthetics_predictor  # noqa: F401
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoAestheticsFilter(Filter):
    """Filter to keep data samples with aesthetics scores for specified frames
    in the videos within a specific range.
    """

    def __init__(self,
                 hf_scorer_model='',
                 min_score: ClosedUnitInterval = 0.4,
                 max_score: ClosedUnitInterval = 1.0,
                 frame_sampling_method: str = 'uniform',
                 frame_num: PositiveInt = 3,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_scorer_model: Huggingface model name for the aesthetics
            predictor. By default, we will use
            'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE',
            refer to pypi.org/project/simple-aesthetics-predictor
        :param min_score: Min score for the predicted aesthetics in a video.
        :param max_score: Max score for the predicted aesthetics in a video.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos.
            Should be one of ["all_keyframes", "uniform"].
            The former one extracts all key frames and the latter one extract
            specified number of frames uniformly from the video.
            Default: "uniform" with frame_num=3, considering that the number of
            keyframes can be large while their difference is usually small
            in terms of their aesthetics.
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param any_or_all: Keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param reduce_mode: reduce mode when one sample corresponds to
            multiple frames, must be one of ['avg','max', 'min'].
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """

        super().__init__(*args, **kwargs)
        if hf_scorer_model == '':
            hf_scorer_model = \
                'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'
        self.min_score = min_score
        self.max_score = max_score

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        if reduce_mode not in ['avg', 'max', 'min']:
            raise ValueError(f'Reduce mode [{reduce_mode}] is not supported. '
                             f'Can only be one of ["avg", "max", "min"].')
        self.any = (any_or_all == 'any')
        self.reduce_mode = reduce_mode

        self.model_key = prepare_model(
            model_type='simple_aesthetics',
            pretrained_model_name_or_path=hf_scorer_model)
        # the original score predicted by laion-ai's scorer is within [0, 10]
        self.need_normalized_by_ten = ('shunk031/aesthetics-predictor'
                                       in hf_scorer_model)
        self._accelerator = 'cuda'
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f'-{frame_sampling_method}' + \
            ('' if frame_sampling_method == 'all_keyframes'
             else f'-{frame_num}')

    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_frames_aesthetics_score in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = (
                np.array([], dtype=np.float64))
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        aesthetics_scores = []
        for key, video in videos.items():
            sampled_frames_key = key + self.sampled_frames_key_suffix
            if video is None:
                continue
            elif context and sampled_frames_key in sample[Fields.context]:
                # sampled frames can be found in the context
                frames = sample[Fields.context][sampled_frames_key]
            else:
                # extract frame images
                if self.frame_sampling_method == 'all_keyframes':
                    frames = extract_key_frames(video)
                elif self.frame_sampling_method == 'uniform':
                    frames = extract_video_frames_uniformly(
                        video, self.frame_num)
                else:
                    frames = []

                # store the sampled frames in the context
                if context:
                    sample[Fields.context][sampled_frames_key] = frames
            frame_images = [frame.to_image() for frame in frames]

            # compute aesthetics_scores
            model, processor = get_model(self.model_key, rank=rank)
            inputs = processor(images=frame_images,
                               return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            if self.need_normalized_by_ten:
                aesthetics_score = (outputs.logits / 10.0).detach().cpu()
            else:
                aesthetics_score = outputs.logits.detach().cpu()

            if self.reduce_mode == 'avg':
                aesthetics_score = float(aesthetics_score.mean())
            elif self.reduce_mode == 'max':
                aesthetics_score = float(aesthetics_score.max())
            else:
                aesthetics_score = float(aesthetics_score.min())
            aesthetics_scores.append(aesthetics_score)

        logger.debug(f'aesthetics_score: {aesthetics_scores}')

        sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = (
            aesthetics_scores)

        if not context:
            for vid_key in videos:
                videos[vid_key].close()

        return sample

    def process(self, sample):
        aesthetics_scores = (
            sample)[Fields.stats][StatsKeys.video_frames_aesthetics_score]
        if len(aesthetics_scores) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= aesthetics_score <= self.max_score
            for aesthetics_score in aesthetics_scores
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
