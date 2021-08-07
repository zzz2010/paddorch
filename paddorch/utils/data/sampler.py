from . import BatchSampler
from paddorch.utils.data import IterableDataset
import paddorch as torch
import math
import numpy as np
from paddle.io import RandomSampler,Sampler

class WeightedRandomSampler(BatchSampler):
    def __init__(self, weights,batch_size, num_samples=None, replacement=True):
        if num_samples is None:
            num_samples=len(weights)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype="float32")
        self.num_samples = num_samples
        self.replacement = replacement
        self.batch_size= batch_size

    def __iter__(self):
        batch_ids=[torch.multinomial(self.weights, self.batch_size, self.replacement)  for _ in range(self.num_samples//self.batch_size)]

        return iter( batch_ids)

    def __len__(self):
        return self.num_samples



class SequentialSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(range(len(self)))


class _InfiniteIterableSampler(object):
    def __init__(self, dataset, batch_size=1):
        assert isinstance(
            dataset, IterableDataset
        ), "dataset should be an instance of paddle.io.IterableDataset"
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield [None] * self.batch_size


class DistributedBatchSampler(BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    In such case, each process can pass a DistributedBatchSampler instance
    as a DataLoader sampler, and load a subset of the original dataset that
    is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset(paddle.io.Dataset): this could be a `paddle.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_size(int): sample indice number in a mini-batch indices.
        num_replicas(int, optional): porcess number in distributed training.
            If :attr:`num_replicas` is None, :attr:`num_replicas` will be
            retrieved from :code:`paddle.fluid.dygraph.parallel.ParallenEnv`.
            Default None.
        rank(int, optional): the rank of the current process among :attr:`num_replicas`
            processes. If :attr:`rank` is None, :attr:`rank` is retrieved from
            :code:`paddle.fluid.dygraph.parallel.ParallenEnv`. Default None.
        shuffle(bool): whther to shuffle indices order before genrating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False

    Examples:
        .. code-block:: python

            import numpy as np

            from paddle.io import Dataset, DistributedBatchSampler

            # init with dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int32')
                    return image, label

                def __len__(self):
                    return self.num_samples

            dataset = RandomDataset(100)
            sampler = DistributedBatchSampler(dataset, batch_size=64)

            for data in sampler:
                # do something
                break
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
            "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                                      self.local_rank * last_local_batch_size:(
                                                                                      self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.

        Arguments:
            epoch (int): Epoch number.

        Examples:
            .. code-block:: python

                import numpy as np

                from paddle.io import Dataset, DistributedBatchSampler

                # init with dataset
                class RandomDataset(Dataset):
                    def __init__(self, num_samples):
                        self.num_samples = num_samples

                    def __getitem__(self, idx):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int32')
                        return image, label

                    def __len__(self):
                        return self.num_samples

                dataset = RandomDataset(100)
                sampler = DistributedBatchSampler(dataset, batch_size=64)

                for epoch in range(10):
                    sampler.set_epoch(epoch)
        """
        self.epoch = epoch
