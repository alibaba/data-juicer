from .base_op import OPERATORS


def load_ops(process_list, text_key='text'):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param text_key: the key name of field that stores sample texts to
        be processed
    :return: The op instance list.
    """
    ops = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))

    return ops
