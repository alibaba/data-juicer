from typing import Any

from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import HashKeys

from ..base_op import Filter

with AvailabilityChecking(['redis'],
                          op_name='ray_xxx_deduplicator',
                          requires_type='dist'):
    try:
        import redis
    except Exception:
        redis = Any


class RayBasicDeduplicator(Filter):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: PositiveInt = 6380,
                 *args,
                 **kwargs):
        """
        Initialization.
        :param redis_host: the hostname of redis server
        :param redis_port: the port of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_host = redis_host
        self.redis_port = redis_port
        # TODO: add a barrier to ensure that flushdb is performed before
        # the operator is called
        r = redis.StrictRedis(host=self.redis_host, port=self.redis_port, db=0)
        r.flushdb(0)

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats(self, sample, context=False):
        # init redis client
        r = redis.StrictRedis(host=self.redis_host, port=self.redis_port, db=0)
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_duplicate] = r.setnx(md5_value, 1)
        return sample

    def process(self, sample):
        return sample[HashKeys.is_duplicate]
