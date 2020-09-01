from . import BatchSampler
import paddorch as torch

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