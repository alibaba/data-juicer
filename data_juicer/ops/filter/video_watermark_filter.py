import numpy as np
from jsonargparse.typing import ClosedUnitInterval, PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

OP_NAME = 'video_watermark_filter'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):

    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling watermark detection in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoWatermarkFilter(Filter):
    """
        Filter to keep samples whose videos have no watermark with high
        probability.
    """

    def __init__(self,
                 hf_watermark_model='amrul-hzz/watermark_detector',
                 prob_threshold: ClosedUnitInterval = 0.8,
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 reduce_mode: str = 'avg',
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_watermark_model: watermark detection model name on
            huggingface.
        :param prob_threshold: the predicted watermark probability threshold
            for samples. range from 0 to 1. Samples with watermark probability
            less than this threshold will be kept.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos.
            Should be one of ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param reduce_mode: reduce mode for multiple sampled video frames.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.prob_threshold = prob_threshold
        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')
        if reduce_mode not in ['avg', 'max', 'min']:
            raise ValueError(f'Reduce mode [{reduce_mode}] is not supported. '
                             f'Can only be one of ["avg", "max", "min"].')
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=hf_watermark_model)
        self._accelerator = 'cuda'
        self.reduce_mode = reduce_mode
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f'-{frame_sampling_method}' + \
            ('' if frame_sampling_method == 'all_keyframes'
             else f'-{frame_num}')

    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_watermark_prob in sample[Fields.stats]:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_watermark_prob] = np.array(
                [], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        watermark_probs = []
        model, processor = get_model(self.model_key, rank=rank)

        for video_key, video in videos.items():
            sampled_frames_key = video_key + self.sampled_frames_key_suffix

            # extract frame images
            if context and sampled_frames_key in sample[Fields.context]:
                frames = sample[Fields.context][sampled_frames_key]
            else:
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
            inputs = processor(images=frame_images, return_tensors='pt')
            inputs = inputs.to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits
            cur_probs = [probs[1] for probs in torch.softmax(logits, dim=-1)]
            cur_probs = torch.Tensor(cur_probs)

            if self.reduce_mode == 'avg':
                watermark_probs.append(cur_probs.mean())
            elif self.reduce_mode == 'max':
                watermark_probs.append(cur_probs.max())
            else:
                watermark_probs.append(cur_probs.min())

        sample[Fields.stats][StatsKeys.video_watermark_prob] = watermark_probs

        if not context:
            for vid_key in videos:
                videos[vid_key].close()

        return sample

    def process(self, sample, rank=None):
        itm_probs = sample[Fields.stats][StatsKeys.video_watermark_prob]
        if len(itm_probs) <= 0:
            return True

        keep_bools = np.array(
            [itm_prob < self.prob_threshold for itm_prob in itm_probs])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
