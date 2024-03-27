import numpy as np
from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from PIL import ImageOps

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (SpecialTokens, extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

OP_NAME = 'video_frames_text_similarity_filter'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):

    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoFramesTextSimilarityFilter(Filter):
    """Filter to keep samples those similarities between sampled video frame
    images and text within a specific range."""

    def __init__(self,
                 hf_clip='openai/clip-vit-base-patch32',
                 min_score: ClosedUnitInterval = 0.1,
                 max_score: ClosedUnitInterval = 1.0,
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the similarity between frame image and text. It's kind of
            language-related. For example, for Chinese datasets, ChineseCLIP
            might be a better choice.
        :param min_score: the min similarity to keep samples.
        :param max_score: the max similarity to keep samples.
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
        :param horizontal_flip: flip frame image horizontally (left to right).
        :param vertical_flip: flip frame image vertically (top to bottom).
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param reduce_mode: reduce mode when one text corresponds to
            multiple video frame images in a chunk.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
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
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_clip)
        self._accelerator = 'cuda'
        self.reduce_mode = reduce_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f'-{frame_sampling_method}' + \
            ('' if frame_sampling_method == 'all_keyframes'
             else f'-{frame_num}')

    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_frames_text_matching_score in sample[Fields.stats]:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][
                StatsKeys.video_frames_text_matching_score] = np.array(
                    [], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        text = sample[self.text_key]
        offset = 0
        similarity = []
        model, processor = get_model(self.model_key, rank=rank)

        for chunk in text.split(SpecialTokens.eoc):
            count = chunk.count(SpecialTokens.video)

            # no video or no text
            if count == 0 or len(chunk) == 0:
                continue
            else:
                text_chunk = remove_special_tokens(chunk)
                video_frame_images_chunk = []
                for video_key in loaded_video_keys[offset:offset + count]:
                    video = videos[video_key]
                    sampled_frames_key = video_key + \
                        self.sampled_frames_key_suffix

                    # extract frame images
                    if context and sampled_frames_key in sample[
                            Fields.context]:
                        # context hit
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
                    for image in frame_images:
                        if self.horizontal_flip:
                            image = ImageOps.mirror(image)
                        if self.vertical_flip:
                            image = ImageOps.flip(image)
                        video_frame_images_chunk.append(image)

                inputs = processor(text=text_chunk,
                                   images=video_frame_images_chunk,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=model.config.text_config.
                                   max_position_embeddings,
                                   padding=True).to(model.device)

                outputs = model(**inputs)
                chunk_logits = outputs.logits_per_text.detach().cpu() / 100.0

                if self.reduce_mode == 'avg':
                    chunk_similarity = chunk_logits.mean()
                elif self.reduce_mode == 'max':
                    chunk_similarity = chunk_logits.max()
                else:
                    chunk_similarity = chunk_logits.min()

                similarity.append(float(chunk_similarity))
            offset += count
        sample[Fields.stats][
            StatsKeys.video_frames_text_matching_score] = similarity

        if not context:
            for vid_key in videos:
                videos[vid_key].close()

        return sample

    def process(self, sample, rank=None):
        similarity = sample[Fields.stats][
            StatsKeys.video_frames_text_matching_score]
        if len(similarity) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= sim_value <= self.max_score
            for sim_value in similarity
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
