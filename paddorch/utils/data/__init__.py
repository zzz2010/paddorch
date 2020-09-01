from paddle import fluid
from paddle.fluid.io import  Dataset,BatchSampler
import numpy as np

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
        return  [np.stack(batch, axis=0)]
    if isinstance(sample, int):
        return np.array(batch)

    if isinstance(sample, fluid.core.VarBase):
        return  [fluid.layers.stack(batch, axis=0)]

    # batch each field
    slots = []
    for sub_batch in list(zip(*batch)):
        slots.append(default_collate_fn(sub_batch))
    return slots


def DataLoader(dataset ,  batch_size ,
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