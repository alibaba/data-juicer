import copy
import inspect
import io
import os

import zstandard as zstd
from loguru import logger

DEFAULT_PREFIX = '__dj__'


class Fields(object):
    stats = DEFAULT_PREFIX + 'stats__'
    meta = DEFAULT_PREFIX + 'meta__'
    context = DEFAULT_PREFIX + 'context__'
    suffix = DEFAULT_PREFIX + 'suffix__'

    # video_frame_tags
    video_frame_tags = DEFAULT_PREFIX + 'video_frame_tags__'
    video_audio_tags = DEFAULT_PREFIX + 'video_audio_tags__'

    # the name of diretory to store the produced multimodal data
    multimodal_data_output_dir = DEFAULT_PREFIX + 'produced_data__'


class StatsKeysMeta(type):
    """
    a helper class to track the mapping from OP's name to its used stats_keys

    e.g., # once the AlphanumericFilter's compute_stats method has been called
    res = TrackingDescriptor.get_access_log()
    print(res) # {"AlphanumericFilter": ["alnum_ratio", "alpha_token_ratio"]}
    """
    _accessed_by = {}

    def __getattr__(cls, attr):
        caller_class = inspect.currentframe().f_back.f_globals['__name__']
        # no need to track the parent classes
        caller_class = caller_class.split('.')[-1]
        stat_key = getattr(cls._constants_class, attr)
        if caller_class not in cls._accessed_by:
            cls._accessed_by[caller_class] = set()
        if stat_key not in cls._accessed_by[caller_class]:
            cls._accessed_by[caller_class].add(stat_key)
        return stat_key

    def get_access_log(cls, dj_cfg=None):
        if cls._accessed_by:
            return cls._accessed_by
        elif dj_cfg:
            tmp_dj_cfg = copy.deepcopy(dj_cfg)
            # the access has been skipped due to the use of cache
            # we will using a temp data sample to get the access log
            if os.path.exists(dj_cfg.dataset_path) and \
                    ('jsonl' in dj_cfg.dataset_path or
                     'jsonl.zst' in dj_cfg.dataset_path):
                logger.info(
                    'Begin to track the usage of ops with a dummy data sample')

                # load the first line as tmp_data
                tmp_f_name = None
                first_line = None
                if 'jsonl.zst' in dj_cfg.dataset_path:
                    tmp_f_name = dj_cfg.dataset_path. \
                        replace('.jsonl.zst', '.tmp.jsonl')
                    # Open the file in binary mode and
                    # create a Zstandard decompression context
                    with open(dj_cfg.dataset_path, 'rb') as compressed_file:
                        dctx = zstd.ZstdDecompressor()
                        # Create a stream reader for the file and decode the
                        # first line
                        with dctx.stream_reader(compressed_file) as reader:
                            text_stream = io.TextIOWrapper(reader,
                                                           encoding='utf-8')
                            first_line = text_stream.readline()
                elif 'jsonl' in dj_cfg.dataset_path:
                    tmp_f_name = dj_cfg.dataset_path. \
                        replace('.jsonl', '.tmp.jsonl')
                    with open(dj_cfg.dataset_path, 'r') as orig_file:
                        first_line = orig_file.readline()

                assert tmp_f_name is not None and first_line is not None, \
                    'error when loading the first line, when ' \
                    f'dj_cfg.dataset_path={dj_cfg.dataset_path}'

                with open(tmp_f_name, 'w') as tmp_file:
                    tmp_file.write(first_line)

                tmp_dj_cfg.dataset_path = tmp_f_name
                tmp_dj_cfg.use_cache = False
                tmp_dj_cfg.use_checkpoint = False

                from data_juicer.core import Analyser
                tmp_analyzer = Analyser(tmp_dj_cfg)
                # do not overwrite the true analysis results
                tmp_analyzer.run(skip_export=True)

                os.remove(tmp_f_name)
            else:
                raise NotImplementedError(
                    f'For now, the dummy data is supported for only jsonl type'
                    f'. Please check your config as {dj_cfg.dataset_path} is '
                    f'either not existed or in jsonl type.')

        return cls._accessed_by


class StatsKeysConstant(object):
    # text
    alpha_token_ratio = 'alpha_token_ratio'
    alnum_ratio = 'alnum_ratio'
    avg_line_length = 'avg_line_length'
    char_rep_ratio = 'char_rep_ratio'
    flagged_words_ratio = 'flagged_words_ratio'
    lang = 'lang'
    lang_score = 'lang_score'
    max_line_length = 'max_line_length'
    perplexity = 'perplexity'
    special_char_ratio = 'special_char_ratio'
    stopwords_ratio = 'stopwords_ratio'
    text_len = 'text_len'
    num_action = 'num_action'
    num_dependency_edges = 'num_dependency_edges'
    num_token = 'num_token'
    num_words = 'num_words'
    word_rep_ratio = 'word_rep_ratio'

    # image
    aspect_ratios = 'aspect_ratios'
    image_width = 'image_width'
    image_height = 'image_height'
    image_sizes = 'image_sizes'
    face_ratios = 'face_ratios'
    face_detections = 'face_detections'
    image_aesthetics_scores = 'image_aesthetics_scores'
    image_nsfw_score = 'image_nsfw_score'
    image_watermark_prob = 'image_watermark_prob'

    # audios
    audio_duration = 'audio_duration'
    audio_nmf_snr = 'audio_nmf_snr'
    audio_sizes = 'audio_sizes'

    # videos
    video_duration = 'video_duration'
    video_aspect_ratios = 'video_aspect_ratios'
    video_width = 'video_width'
    video_height = 'video_height'
    video_ocr_area_ratio = 'video_ocr_area_ratio'
    video_aesthetic_score = 'video_aesthetic_score'
    video_frames_aesthetics_score = 'video_frames_aesthetics_score'
    video_motion_score = 'video_motion_score'
    video_nsfw_score = 'video_nsfw_score'
    video_watermark_prob = 'video_watermark_prob'

    # multimodal
    # image-text
    image_text_similarity = 'image_text_similarity'
    image_text_matching_score = 'image_text_matching_score'
    phrase_grounding_recall = 'phrase_grounding_recall'

    # video-text
    video_frames_text_matching_score = 'video_frames_text_matching_score'


class StatsKeys(object, metaclass=StatsKeysMeta):
    _constants_class = StatsKeysConstant


class HashKeys(object):
    hash = DEFAULT_PREFIX + 'hash'
    minhash = DEFAULT_PREFIX + 'minhash'
    simhash = DEFAULT_PREFIX + 'simhash'

    # image
    imagehash = DEFAULT_PREFIX + 'imagehash'

    # video
    videohash = DEFAULT_PREFIX + 'videohash'

    # duplicate flag
    is_duplicate = DEFAULT_PREFIX + 'is_duplicate'


class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'
    refined_words = DEFAULT_PREFIX + 'refined_words'

    # image
    loaded_images = DEFAULT_PREFIX + 'loaded_images'  # Image

    # audios
    loaded_audios = DEFAULT_PREFIX + 'loaded_audios'  # (data, sampling_rate)

    # videos
    # InputContainer from av.
    # Key: {video_path}
    loaded_videos = DEFAULT_PREFIX + 'loaded_videos'
    # sampled frames.
    # Key: {video_path}-{frame_sampling_method}[-{frame_num}]
    #   {frame_num} is only used when {frame_sampling_method} is "uniform"
    sampled_frames = DEFAULT_PREFIX + 'sampled_frames'
