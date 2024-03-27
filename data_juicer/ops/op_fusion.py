from typing import List

from loguru import logger

from data_juicer.utils.constant import Fields, InterVars
from data_juicer.utils.registry import Registry

from .base_op import Filter

# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# audios
LOADED_AUDIOS = Registry(InterVars.loaded_audios)

# videos
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_IMAGES, LOADED_VIDEOS]


def fuse_operators(process_list, ops):
    """
    Fuse the input ops list and return the fused ops list.

    :param process_list: the list of original process definition, including op
        names and args.
    :param ops: the corresponding list of op objects.
    :return: a list of fused op objects.
    """
    # detect filter groups and try to fuse them
    fused_op_def = []
    fused_ops = []
    filter_group = []
    in_group = False
    for process, op in zip(process_list, ops):
        if isinstance(op, Filter):
            if not in_group:
                in_group = True
            filter_group.append((process, op))
        elif in_group:
            # got a filter group, try to fuse them
            fused_group_def, fused_group = fuse_filter_group(filter_group)
            fused_op_def.extend(fused_group_def)
            fused_ops.extend(fused_group)
            filter_group = []
            in_group = False
            # and add the current non-filter op into fused_ops
            fused_op_def.append(process)
            fused_ops.append(op)
        else:  # not a filter and not in a filter group, skip
            fused_op_def.append(process)
            fused_ops.append(op)
    if in_group and len(filter_group) > 0:
        # the final filter group, try to fuse them
        fused_group_def, fused_group = fuse_filter_group(filter_group)
        fused_op_def.extend(fused_group_def)
        fused_ops.extend(fused_group)
    return fused_op_def, fused_ops


def fuse_filter_group(original_filter_group):
    """
    Fuse single filter group and return the fused filter group.

    :param original_filter_group: the original filter group, including op
        definitions and objects.
    :return: the fused definitions and objects of the input filter group.
    """
    fused_group_def = []
    fused_group = []
    all_intermediate_vars = ALL_INTER_VARS
    all_fused_filters = {
        inter_vars: []
        for inter_vars in all_intermediate_vars
    }
    # group these filters by their intermediate vars
    for process, op in original_filter_group:
        op_name, op_args = list(process.items())[0]
        for inter_vars in all_intermediate_vars:
            if op_name in inter_vars.modules:
                all_fused_filters[inter_vars].append((process, op))
                break
        else:
            # first apply other filters to decrease the number of samples, so
            # we add them into the fused_group list directly
            fused_group_def.append(process)
            fused_group.append(op)

    # try to fuse ops for each type of intermediate vars
    for inter_vars in all_intermediate_vars:
        inter_vars_filter = all_fused_filters[inter_vars]
        if len(inter_vars_filter) == 0:
            # no ops include this type of intermediate var
            pass
        elif len(inter_vars_filter) > 1:
            # more than 1 ops share the same intermediate var, try to fuse them
            defs, ops = zip(*inter_vars_filter)
            # new definition: new name and a definition list of fused op list
            fused_filter_def = {
                'OpFusion:(%s)' % ','.join([
                    list(process.items())[0][0] for process in defs
                ]):
                list(defs)
            }
            logger.info(f'Ops are fused into one op '
                        f'{list(fused_filter_def.keys())[0]}.')
            # use these ops to create a FusedFilter object, and add the fused
            # definition and op into the fused group
            fused_filter = FusedFilter(ops)
            fused_group_def.append(fused_filter_def)
            fused_group.append(fused_filter)
        else:
            # only 1 op for this type of intermediate var, add it to the fused
            # group directly without fusion
            fused_group_def.append(inter_vars_filter[0][0])
            fused_group.append(inter_vars_filter[0][1])

    return fused_group_def, fused_group


class FusedFilter(Filter):
    """A fused operator for filters."""

    def __init__(self, fused_filters: List):
        """
        Initialization method.

        :param fused_filters: a list of filters to be fused.
        """
        super().__init__()
        self.fused_filters = fused_filters
        # set _accelerator to 'cuda' if there exists any ops whose _accelerator
        # is 'cuda'
        accelerator_methods = set(
            [op._accelerator for op in self.fused_filters])
        if 'cuda' in accelerator_methods:
            self._accelerator = 'cuda'

    def compute_stats(self, sample, rank=None):
        import av

        # context for the intermediate vars
        sample[Fields.context] = {}
        for op in self.fused_filters:
            # open the context for these fused ops
            if op._accelerator == 'cuda':
                sample = op.compute_stats(sample, rank=rank, context=True)
            else:
                sample = op.compute_stats(sample, context=True)
        # clean up the contexts after processing
        # check if there are containers that need to be closed
        for context_key in sample[Fields.context]:
            if isinstance(sample[Fields.context][context_key],
                          av.container.InputContainer):
                sample[Fields.context][context_key].close()
        _ = sample.pop(Fields.context)
        return sample

    def process(self, sample):
        # Only return True when all filters return True
        for op in self.fused_filters:
            if not op.process(sample):
                return False
        return True
