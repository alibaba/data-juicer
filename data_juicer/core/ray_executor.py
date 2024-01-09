from loguru import logger

from data_juicer.config import init_configs
from data_juicer.ops import Filter, Mapper, load_ops
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray
    import ray.data as rd


class RayExecutor:
    """
    Executor based on Ray [Experimental].

    Run Data-Juicer data processing in a distributed cluster.
        1. Only support Filter and Mapper operators for now.
        2. Only support loading `.json` files.
        2. Advanced functions such as checkpoint, tracer are not supported.
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

    def run(self, load_data_np=None):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. load data
        logger.info('Loading dataset with Ray...')
        dataset = rd.read_json(self.cfg.dataset_path)

        # 2. extract processes
        logger.info('Preparing process operators...')
        self.process_list, self.ops = load_ops(self.cfg.process,
                                               self.cfg.op_fusion)

        # 3. data process
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        if Fields.stats not in dataset.columns(fetch_if_missing=False):
            logger.info(f'columns {dataset.columns(fetch_if_missing=False)}')
            dataset = dataset.add_column(
                Fields.stats, lambda df: [{} for _ in range(len(df))])
        logger.info('Processing data...')
        for op_cfg, op in zip(self.process_list, self.ops):
            op_name, _ = list(op_cfg.items())[0]
            try:
                if isinstance(op, Mapper):
                    dataset = dataset.map(op.process)
                elif isinstance(op, Filter):
                    dataset = dataset.map(op.compute_stats)
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

            # clean up cache files and record processed ops
            logger.info(f'Op [{op_name}] Done. Left '
                        f'{dataset.count()} samples.')

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.write_json(self.cfg.export_path, force_ascii=False)
        return dataset
