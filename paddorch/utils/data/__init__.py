from paddle import fluid
from paddle.fluid.io import  Dataset,BatchSampler
import numpy as np
import paddorch
import paddle
from . import  graph

def default_collate_fn(batch):
    """
    Default batch collating function for :code:`fluid.io.DataLoader`,
    batch should be a list of samples, and each sample should be a list
    of fields as follows:

    [[filed1, filed2, ...], [filed1, filed2, ...], ...]

    This default collate function zipped each filed together and stack
    each filed as the batch field as follows:
    [batch_filed1, batch_filed2, ...]
    Args:
        batch(list of list of numpy array): the batch data, each fields
              should be a numpy array, each sample should be a list of
              fileds, and batch should be a list of sample.

    Returns:
        a list of numpy array: collated batch
    """

    sample = batch[0]
    # dataset has only 1 field
    if isinstance(sample, np.ndarray):
        return   np.stack(batch, axis=0)  #[np.stack(batch, axis=0)]
    if isinstance(sample, int) or isinstance(sample, np.int32):
        return  np.array(batch)

    if isinstance(sample, fluid.core.VarBase):
        return  fluid.layers.stack(batch, axis=0)  #[fluid.layers.stack(batch, axis=0)]

    # batch each field
    slots = []
    for sub_batch in list(zip(*batch)):
        slots.append(default_collate_fn(sub_batch))
    return slots


def DataLoader(dataset ,  batch_size=None ,
                           shuffle=False,
                           batch_sampler=None,num_workers=0,collate_fn=None,
                           pin_memory=False,
                           drop_last=False    ,            timeout=0,
                            worker_init_fn=None):

    if collate_fn is None:
        collate_fn=default_collate_fn

    return   fluid.io.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                                 places=[fluid.dygraph.framework._current_expected_place()],
                                 collate_fn=collate_fn,
                           batch_sampler=batch_sampler,
                           use_shared_memory=False,
                           drop_last=drop_last,timeout=timeout,worker_init_fn=worker_init_fn)


class IterableDataset(Dataset):
    """
    An abstract class to encapsulate methods and behaviors of iterable datasets.
    All datasets in iterable-style (can only get sample one by one sequentially, like
    a Python iterator) should be a subclass of `paddle.io.IterableDataset`. All subclasses should
    implement following methods:
    :code:`__iter__`: yield sample sequentially. This method is required by reading dataset sample in :code:`paddle.io.DataLoader`.
    .. note::
        do not implement :code:`__getitem__` and :code:`__len__` in IterableDataset, should not be called either.
    see :code:`paddle.io.DataLoader`.
    Examples:

        .. code-block:: python
            import numpy as np
            from paddle.io import Dataset

            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __iter__(self):
                    for i in range(self.num_samples):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int32')
                        yield image, label

            dataset = RandomDataset(10)
            for img, lbl in dataset:
                print(img, lbl)
    When :attr:`num_workers > 0`, each worker has a different copy of the dataset object and
    will yield whole dataset samples, which means samples in dataset will be repeated in
    :attr:`num_workers` times. If it is required for each sample to yield only once, there
    are two methods to configure different copy in each worker process to avoid duplicate data
    among workers as follows. In both the methods, worker information that can be getted in
    a worker process by `paddle.io.get_worker_info` will be needed.
    Example 1: splitting data copy in each worker in :code:`__iter__`
        .. code-block:: python
            import math
            import numpy as np
            import paddle.fluid as fluid
            from paddle.io import IterableDataset, DataLoader, get_worker_info
            class SplitedIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
                def __iter__(self):
                    worker_info = get_worker_info()
                    if worker_info is None:
                        iter_start = self.start
                        iter_end = self.end
                    else:
                        per_worker = int(
                            math.ceil((self.end - self.start) / float(
                                worker_info.num_workers)))
                        worker_id = worker_info.id
                        iter_start = self.start + worker_id * per_worker
                        iter_end = min(iter_start + per_worker, self.end)
                    for i in range(iter_start, iter_end):
                        yield np.array([i])
            place = fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                dataset = SplitedIterableDataset(start=2, end=9)
                dataloader = DataLoader(
                    dataset,
                    places=place,
                    num_workers=2,
                    batch_size=1,
                    drop_last=True)
                print(list(dataloader))
                # outputs: [2, 5, 3, 6, 4, 7]
    Example 2: splitting data copy in each worker by :code:`worker_init_fn`
        .. code-block:: python
            import math
            import numpy as np
            import paddle.fluid as fluid
            from paddle.io import IterableDataset, DataLoader, get_worker_info
            class RangeIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
                def __iter__(self):
                    for i in range(self.start, self.end):
                        yield np.array([i])
            place = fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                dataset = RangeIterableDataset(start=2, end=9)
                def worker_init_fn(worker_id):
                    worker_info = get_worker_info()
                    dataset = worker_info.dataset
                    start = dataset.start
                    end = dataset.end
                    num_per_worker = int(
                        math.ceil((end - start) / float(worker_info.num_workers)))
                    worker_id = worker_info.id
                    dataset.start = start + worker_id * num_per_worker
                    dataset.end = min(dataset.start + num_per_worker, end)
                dataloader = DataLoader(
                    dataset,
                    places=place,
                    num_workers=2,
                    batch_size=1,
                    drop_last=True,
                    worker_init_fn=worker_init_fn)
                print(list(dataloader))
                # outputs: [2, 5, 3, 6, 4, 7]
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class " \
                                  "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                           "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                           "{}".format('__len__', self.__class__.__name__))


class TensorDataset(Dataset):
    """
    Dataset defined by a list of tensors.
    Each tensor should be in shape of [N, ...], while N is the sample number,
    and ecah tensor contains a field of sample, :code:`TensorDataset` retrieve
    each sample by indexing tensors in the 1st dimension.
    Args:
        tensors(list of Tensor): tensors with same shape in the 1st dimension.
    Returns:
        Dataset: a Dataset instance wrapping tensors.
    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.io import TensorDataset
            paddle.disable_static()
            input_np = np.random.random([2, 3, 4]).astype('float32')
            input = paddle.to_tensor(input_np)
            label_np = np.random.random([2, 1]).astype('int32')
            label = paddle.to_tensor(label_np)
            dataset = TensorDataset([input, label])
            for i in range(len(dataset)):
                input, label = dataset[i]
                print(input, label)
    """

    def __init__(self, tensors):
        if not fluid.framework.in_dygraph_mode():
            raise RuntimeError(
                "TensorDataset con only be used in imperative mode")
        assert all([tensor.shape[0] == tensors[0].shape[0] for tensor in tensors]), \
            "tensors not have same shape of the 1st dimension"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


def get_worker_info():
    return paddle.fluid.io.get_worker_info()


class Subset(Dataset ):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset
    indices: list

    def __init__(self, dataset: Dataset, indices) -> None:
        self.dataset = dataset
        if isinstance(indices,np.ndarray):
            indices=indices.tolist()
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)