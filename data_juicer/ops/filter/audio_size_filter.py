import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import get_file_size, size_to_bytes

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('audio_size_filter')
class AudioSizeFilter(Filter):
    """Keep data samples whose audio size (in bytes/kb/MB/...) within a
    specific range.
    """

    def __init__(self,
                 min_size: str = '0',
                 max_size: str = '1TB',
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_size: The min audio size to keep samples.  set to be "0" by
        default for no size constraint
        :param max_size: The max audio size to keep samples.  set to be
        "1Tb" by default, an approximate for un-limited case
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all audios. 'any': keep this sample if any audios meet the
            condition. 'all': keep this sample only if all audios meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_size = size_to_bytes(min_size)
        self.max_size = size_to_bytes(max_size)
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.audio_sizes in sample[Fields.stats]:
            return sample

        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_sizes] = np.array(
                [], dtype=np.float64)
            return sample

        # for size calculation, no need to load audios into memory
        sample[Fields.stats][StatsKeys.audio_sizes] = [
            get_file_size(img_path) for img_path in sample[self.audio_key]
        ]

        return sample

    def process(self, sample):
        audio_sizes = sample[Fields.stats][StatsKeys.audio_sizes]
        keep_bools = np.array([
            self.min_size <= audio_size <= self.max_size
            for audio_size in audio_sizes
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
