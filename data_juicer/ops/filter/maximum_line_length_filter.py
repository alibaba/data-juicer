import sys

from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('maximum_line_length_filter')
class MaximumLineLengthFilter(Filter):
    """Filter to keep samples with maximum line length within a specific
    range."""

    def __init__(self,
                 min_len: PositiveInt = 10,
                 max_len: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_len: The min filter length in this op, samples will
            be filtered if their maximum line length is below this
            parameter.
        :param max_len: The max filter length in this op, samples will
            be filtered if their maximum line length exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len

    def compute_stats(self, sample):
        # check if it's computed already
        if StatsKeys.max_line_length in sample[Fields.stats]:
            return sample

        line_lengths = list(map(len, sample[self.text_key].splitlines()))
        sample[Fields.stats][StatsKeys.max_line_length] = max(
            line_lengths) if line_lengths else 0.0
        return sample

    def process(self, sample):
        if self.min_len <= sample[Fields.stats][
                StatsKeys.max_line_length] <= self.max_len:
            return True
        else:
            return False
