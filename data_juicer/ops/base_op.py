import copy

import pandas as pd
import pyarrow as pa

from data_juicer.utils.mm_utils import size_to_bytes
from data_juicer.utils.registry import Registry

OPERATORS = Registry('Operators')


class OP:

    def __init__(self, *args, **kwargs):
        """
        Base class of operators.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        # init data keys
        self.text_key = kwargs.get('text_key', 'text')
        self.image_key = kwargs.get('image_key', 'images')
        self.audio_key = kwargs.get('audio_key', 'audios')
        self.video_key = kwargs.get('video_key', 'videos')

        # whether the model can be accelerated using cuda
        self._accelerator = kwargs.get('accelerator', 'cpu')

        # parameters to determind the number of procs for this op
        self.spec_numprocs = kwargs.get('spec_numprocs', 0)
        self.cpu_required = kwargs.get('cpu_required', 1)
        self.mem_required = kwargs.get('mem_required', 0)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3

        # whether to use actor mode in ray
        self._use_actor = kwargs.get('use_actor', False)

        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_actor(self):
        return self._use_actor

    def remove_extra_parameters(self, param_dict, keys=None):
        """
            at the begining of the init of the mapper op, call
            self.remove_extra_parameters(locals())
            to get the init parameter dict of the op for convenience

        """
        if keys is None:
            param_dict = {
                k: v
                for k, v in param_dict.items() if not k.startswith('_')
            }
            param_dict.pop('self', None)
        else:
            param_dict = {k: v for k, v in param_dict.items() if k not in keys}
        return param_dict

    def add_parameters(self, init_parameter_dict, **extra_param_dict):
        """
            add parameters for each sample, need to keep extra_param_dict
            and init_parameter_dict unchanged.
        """
        related_parameters = copy.deepcopy(init_parameter_dict)
        related_parameters.update(extra_param_dict)
        return related_parameters


def ray_batch_mapper_wrapper(samples, fn):
    samples = samples.to_pandas()
    res = fn(samples)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    return pa.Table.from_pandas(res)


class Mapper(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts data editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Mapper, self).__init__(*args, **kwargs)

        # In default, it's a normal OP instead of batched OP
        self._batched_op = kwargs.get('batched_op', False)

    def process(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def is_batched_op(self):
        return self._batched_op

    def __call__(self, sample):
        """
        Make the class callable to enable ray actor usage
        """
        if self.is_batched_op():
            # same logic as ray_batch_mapper_wrapper
            samples = sample.to_pandas()
            res = self.process(samples)
            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame(res)
            return pa.Table.from_pandas(res)
        else:
            return self.process(sample)


class Filter(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that removes specific info.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Filter, self).__init__(*args, **kwargs)

        from data_juicer.core.data import wrap_func_with_nested_access
        self.compute_stats = wrap_func_with_nested_access(self.compute_stats)

    def compute_stats(self, sample, context=False):
        """
        Compute stats for the sample which is used as a metric to decide
        whether to filter this sample.

        :param sample: input sample.
        :param context: whether to store context information of intermediate
            vars in the sample temporarily.
        :return: sample with computed stats
        """
        raise NotImplementedError

    def process(self, sample):
        """
        For sample level, sample --> Boolean.

        :param sample: sample to decide whether to filter
        :return: true for keeping and false for filtering
        """
        raise NotImplementedError

    def __call__(self, sample):
        """
        Make the class callable to enable ray actor usage
        """
        return self.compute_stats(sample)


class Deduplicator(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts deduplication.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Deduplicator, self).__init__(*args, **kwargs)

        from data_juicer.core.data import wrap_func_with_nested_access
        self.compute_hash = wrap_func_with_nested_access(self.compute_hash)

    def compute_hash(self, sample):
        """
        Compute hash values for the sample.

        :param sample: input sample
        :return: sample with computed hash value.
        """
        raise NotImplementedError

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        raise NotImplementedError


class Selector(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts selection in dataset-level.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Selector, self).__init__(*args, **kwargs)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: selected dataset.
        """
        raise NotImplementedError
