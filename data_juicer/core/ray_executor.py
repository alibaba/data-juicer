import math
import os
import subprocess
import time
from functools import partial

import pandas as pd
import psutil
import pyarrow as pa
import torch
from loguru import logger

from data_juicer import cuda_device_count, use_cuda
from data_juicer.config import init_configs
from data_juicer.ops import Filter, Mapper, load_ops
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray
    import ray.data as rd
    from ray.data import ActorPoolStrategy

from data_juicer.ops.base_op import OPERATORS


def is_valid_path(item, dataset_dir):
    full_path = os.path.abspath(os.path.join(dataset_dir, item))
    return os.path.exists(full_path)


def convert_to_absolute_paths(dict_with_paths, dataset_dir):
    for key, value in dict_with_paths.items():
        if isinstance(value, list):
            dict_with_paths[key] = [
                os.path.abspath(os.path.join(dataset_dir, item))
                if isinstance(item, str) and is_valid_path(dataset_dir, item)
                else item for item in value
            ]
        elif isinstance(value, str):
            dict_with_paths[key] = os.path.abspath(
                os.path.join(
                    dataset_dir,
                    value)) if isinstance(value, str) and is_valid_path(
                        value, dataset_dir) else value
    return dict_with_paths


def set_dataset_to_absolute_path(dataset, dataset_path):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    dataset_dir = os.path.dirname(dataset_path)
    dataset = dataset.map(
        lambda item: convert_to_absolute_paths(item, dataset_dir))
    print(f"transfer {dataset.count()} sample's paths")
    return dataset


def ray_batch_mapper_wrapper(samples, fn):
    samples = samples.to_pandas()
    res = fn(samples)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    return pa.Table.from_pandas(res)


class RayExecutor:
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = init_configs() if cfg is None else cfg

        self.work_dir = self.cfg.work_dir

        self.ops = None
        # init ray
        logger.info('Initing Ray ...')
        ray.init(self.cfg.ray_address)
        self.process_list = self.cfg.process

    def get_min_cuda_memory(self):
        # get cuda memory info using "nvidia-smi" command
        min_cuda_memory = torch.cuda.get_device_properties(
            0).total_memory / 1024**2
        nvidia_smi_output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')
        for line in nvidia_smi_output.strip().split('\n'):
            free_memory = int(line)
            min_cuda_memory = min(min_cuda_memory, free_memory)
        return min_cuda_memory

    def calculate_np(self, op, op_name):
        if self.cfg.np is None:
            self.cfg.np = psutil.cpu_count()
        if use_cuda() and op._accelerator == 'cuda':
            cuda_mem_available = self.get_min_cuda_memory() / 1024
            op_proc = min(
                self.cfg.np,
                math.floor(cuda_mem_available / (op.mem_required + 0.1)) *
                cuda_device_count())
            if op_proc < 1.0:
                logger.warning(
                    f'The required cuda memory:{op.mem_required}GB might '
                    f'be more than the available cuda memory:'
                    f'{cuda_mem_available}GB.'
                    f'This Op [{op_name}] might '
                    f'require more resource to run.')
            op_proc = max(op_proc, 1)
            return op_proc
        else:
            op_proc = self.cfg.np
            cpu_available = psutil.cpu_count()
            mem_available = psutil.virtual_memory().available
            mem_available = mem_available / 1024**3
            op_proc = min(op_proc, math.floor(cpu_available / op.cpu_required))
            op_proc = min(op_proc,
                          math.floor(mem_available / (op.mem_required + 0.1)))
            if op_proc < 1.0:
                logger.warning(
                    f'The required CPU number:{op.cpu_required} '
                    f'and memory:{op.mem_required}GB might '
                    f'be more than the available CPU:{cpu_available} '
                    f'and memory :{mem_available}GB.'
                    f'This Op [{op_name}] might '
                    f'require more resource to run.')
            op_proc = max(op_proc, 1)
            return op_proc

    def get_num_gpus(self, op, op_proc):
        if not use_cuda() or not op._accelerator == 'cuda':
            return 0
        proc_per_gpu = op_proc / cuda_device_count()
        return 1.0 / proc_per_gpu

    def run(self, load_data_np=None):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. load data
        logger.info('Loading dataset with Ray...')
        dataset = rd.read_json(self.cfg.dataset_path)

        # convert all the path in dataset to absolute path
        dataset = set_dataset_to_absolute_path(dataset, self.cfg.dataset_path)
        for items in dataset.iter_rows():
            print('item is:', items)
        # 2. extract processes
        logger.info('Preparing process operators...')
        self.process_list, self.ops = load_ops(self.cfg.process,
                                               self.cfg.op_fusion)

        # 3. data process
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        if Fields.stats not in dataset.columns(fetch_if_missing=False):
            logger.info(f'columns {dataset.columns(fetch_if_missing=False)}')

            def process_batch_arrow(table: pa.Table) -> pa.Table:
                new_column_data = [{} for _ in range(len(table))]
                new_talbe = table.append_column(Fields.stats,
                                                [new_column_data])
                return new_talbe

            dataset = dataset.map_batches(process_batch_arrow,
                                          batch_format='pyarrow')

        logger.info('Processing data...')
        tstart = time.time()
        for op_cfg, op in zip(self.process_list, self.ops):
            num_gpus = 1 if use_cuda() and op._accelerator == 'cuda' else 0
            op_name, op_args = list(op_cfg.items())[0]
            try:
                if isinstance(op, Mapper):
                    if op.is_batched_op():
                        if op.use_actor():
                            op_proc = self.calculate_np(op, op_name)
                            num_gpus = self.get_num_gpus(op, op_proc)
                            op_cls = OPERATORS.modules[op_name]
                            dataset = dataset.map_batches(
                                op_cls,
                                compute=ActorPoolStrategy(),
                                concurrency=op_proc,
                                fn_constructor_kwargs=op_args,
                                batch_format='pyarrow',
                                num_gpus=num_gpus,
                                batch_size=1)
                            # The batch size here is same as in data.py
                        else:
                            dataset = dataset.map_batches(
                                partial(ray_batch_mapper_wrapper,
                                        fn=op.process),
                                batch_format='pyarrow',
                                num_gpus=num_gpus,
                                batch_size=1)
                            # The batch size here is same as in data.py
                    else:
                        if op.use_actor():
                            op_cls = OPERATORS.modules[op_name]
                            op_proc = self.calculate_np(op, op_name)
                            num_gpus = self.get_num_gpus(op, op_proc)
                            dataset = dataset.map(
                                op_cls,
                                compute=ActorPoolStrategy(),
                                concurrency=op_proc,
                                fn_constructor_kwargs=op_args,
                                num_gpus=num_gpus)
                        else:
                            dataset = dataset.map(op.process,
                                                  num_gpus=num_gpus)

                elif isinstance(op, Filter):
                    if op.use_actor():
                        op_cls = OPERATORS.modules[op_name]
                        op_proc = self.calculate_np(op, op_name)
                        num_gpus = self.get_num_gpus(op, op_proc)
                        dataset = dataset.map(op_cls,
                                              compute=ActorPoolStrategy(),
                                              concurrency=self.cfg.np,
                                              fn_constructor_kwargs=op_args,
                                              num_gpus=num_gpus)
                    else:
                        dataset = dataset.map(op.compute_stats,
                                              num_gpus=num_gpus)
                    dataset = dataset.filter(op.process)
                else:
                    logger.error(
                        'Ray executor only support Filter and Mapper OPs for '
                        'now')
                    raise NotImplementedError
            except:  # noqa: E722
                logger.error(f'An error occurred during Op [{op_name}].')
                import traceback
                traceback.print_exc()
                exit(1)

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.write_json(self.cfg.export_path, force_ascii=False)
        tend = time.time()
        logger.info(f'All Ops are done in {"%.3f" % (tend - tstart)}(s).')
        return dataset
