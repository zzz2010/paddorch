# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""sampler
"""

import time
import numpy as np
from . import mp_reader

class Sampler(object):
    """Sampler
    """

    def __init__(self, dataset, batch_size=1, drop_last=False, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        perm = np.arange(0, len(self.dataset))
        if self.shuffle:
            np.random.shuffle(perm)

        batch = []
        for idx in perm:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            length = length // self.batch_size
        else:
            length = (length + self.batch_size - 1) // self.batch_size
        return length


class StreamSampler(object):
    """StreamSampler
    """

    def __init__(self, dataset, batch_size=1, drop_last=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = [i for i in range(self.batch_size)]
        while True:
            yield batch

import warnings
import time
import numpy as np
from collections import namedtuple

import paddle
from paddle.fluid.io import  Dataset

WorkerInfo = namedtuple("WorkerInfo", ["num_workers", "fid"])


class Dataloader(object):
    """Dataloader for loading batch data

    Example:

        .. code-block:: python

            from pgl.utils.data import Dataset
            from pgl.utils.data.dataloader import Dataloader

            class MyDataset(Dataset):
                def __init__(self):
                    self.data = list(range(0, 40))

                def __getitem__(self, idx):
                    return self.data[idx]

                def __len__(self):
                    return len(self.data)

            def collate_fn(batch_examples):
                inputs = np.array(batch_examples, dtype="int32")
                return inputs

            dataset = MyDataset()
            loader = Dataloader(dataset,
                                batch_size=3,
                                drop_last=False,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=collate_fn)

            for batch_data in loader:
                print(batch_data)

    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 drop_last=False,
                 shuffle=False,
                 num_workers=1,
                 collate_fn=None,
                 buf_size=1000,
                 stream_shuffle_size=0,worker_init_fn=None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.buf_size = buf_size
        self.drop_last = drop_last
        self.stream_shuffle_size = stream_shuffle_size

        if self.shuffle and isinstance(self.dataset, StreamDataset):
            warn_msg = "The argument [shuffle] should not be True with StreamDataset. " \
                    "It will be ignored. " \
                    "You might want to set [stream_shuffle_size] with StreamDataset."
            warnings.warn(warn_msg)

        if self.stream_shuffle_size > 0 and self.batch_size >= stream_shuffle_size:
            raise ValueError("stream_shuffle_size must be larger than batch_size," \
                    "but got [stream_shuffle_size=%s] smaller than [batch_size=%s]" \
                    % (self.stream_shuffle_size, self.batch_size))

        if self.stream_shuffle_size > 0 and isinstance(self.dataset, Dataset):
            warn_msg = "[stream_shuffle_size] should not be set with Dataset. " \
                    "It will be ignored. " \
                    "You might want to set [shuffle] with Dataset."
            warnings.warn(warn_msg)

        if self.num_workers < 1:
            raise ValueError("num_workers(default: 1) should be larger than 0, " \
                        "but got [num_workers=%s] < 1." % self.num_workers)

        if isinstance(self.dataset, StreamDataset):  # for stream data
            # generating a iterable sequence for produce batch data without repetition
            self.sampler = StreamSampler(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last)
        else:
            self.sampler = Sampler(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle)

    def __len__(self):
        if not isinstance(self.dataset, StreamDataset):
            return len(self.sampler)
        else:
            raise Exception("StreamDataset has no length")

    def __iter__(self):
        # random seed will be fixed when using multiprocess,
        # so set seed explicitly every time
        np.random.seed(0)
        if self.num_workers == 1:
            r = paddle.reader.buffered(_DataLoaderIter(self, 0), self.buf_size)
        else:
            worker_pool = [
                _DataLoaderIter(self, wid) for wid in range(self.num_workers)
            ]
            workers = mp_reader.multiprocess_reader(
                worker_pool, use_pipe=True, queue_size=1000)
            r = paddle.reader.buffered(workers, self.buf_size)

        for batch in r():
            yield batch

    def __call__(self):
        return self.__iter__()


class _DataLoaderIter(object):
    """Iterable DataLoader Object
    """

    def __init__(self, dataloader, fid=0):
        self.dataset = dataloader.dataset
        self.sampler = dataloader.sampler
        self.collate_fn = dataloader.collate_fn
        self.num_workers = dataloader.num_workers
        self.drop_last = dataloader.drop_last
        self.batch_size = dataloader.batch_size
        self.stream_shuffle_size = dataloader.stream_shuffle_size
        self.fid = fid

    def _data_generator(self):
        for count, indices in enumerate(self.sampler):

            if count % self.num_workers != self.fid:
                continue

            batch_data = [self.dataset[i] for i in indices]

            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def _streamdata_generator(self):
        self._worker_info = WorkerInfo(
            num_workers=self.num_workers, fid=self.fid)
        self.dataset._set_worker_info(self._worker_info)

        dataset = iter(self.dataset)
        for indices in self.sampler:
            batch_data = []
            for _ in indices:
                try:
                    batch_data.append(next(dataset))
                except StopIteration:
                    break

            if len(batch_data) == 0 or (self.drop_last and
                                        len(batch_data) < len(indices)):
                break
                #  raise StopIteration

            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def _stream_shuffle_data_generator(self):
        def _stream_shuffle_index_generator():
            shuffle_size = [i for i in range(self.stream_shuffle_size)]
            while True:
                yield shuffle_size

        def _data_generator():
            dataset = iter(self.dataset)
            for shuffle_size in _stream_shuffle_index_generator():
                shuffle_size_data = []
                for idx in shuffle_size:
                    try:
                        shuffle_size_data.append(next(dataset))
                    except StopIteration:
                        break

                if len(shuffle_size_data) == 0:
                    break

                yield shuffle_size_data

        def _batch_data_generator():
            batch_data = []
            for shuffle_size_data in _data_generator():
                np.random.shuffle(shuffle_size_data)

                for d in shuffle_size_data:
                    batch_data.append(d)
                    if len(batch_data) == self.batch_size:
                        yield batch_data
                        batch_data = []

            if not self.drop_last and len(batch_data) > 0:
                yield batch_data

        self._worker_info = WorkerInfo(
            num_workers=self.num_workers, fid=self.fid)
        self.dataset._set_worker_info(self._worker_info)

        for batch_data in _batch_data_generator():
            if self.collate_fn is not None:
                yield self.collate_fn(batch_data)
            else:
                yield batch_data

    def __iter__(self):
        if isinstance(self.dataset, StreamDataset):
            if self.stream_shuffle_size > 0:
                data_generator = self._stream_shuffle_data_generator
            else:
                data_generator = self._streamdata_generator
        else:
            data_generator = self._data_generator

        for batch_data in data_generator():
            yield batch_data

    def __call__(self):
        return self.__iter__()



class StreamDataset(object):
    """An abstract class represening StreamDataset which has unknown length.
    Generally, all unknown length datasets should subclass it.
    All subclasses should overwrite :code:`__iter__`.
    Examples:
        .. code-block:: python
            from pgl.utils.data import StreamDataset
            class MyStreamDataset(StreamDataset):
                def __init__(self):
                    self.data = list(range(0, 40))
                    self.count = 0
                def __iter__(self):
                     for data in self.dataset:
                        self.count += 1
                        if self.count % self._worker_info.num_workers != self._worker_info.fid:
                            continue
                        # do something (like parse data)  of your data
                        time.sleep(0.1)
                        yield data
    """

    def __iter__(self):
        raise NotImplementedError

    def _set_worker_info(self, worker_info):
        self._worker_info = worker_info