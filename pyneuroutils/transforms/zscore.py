import scipy.stats as stats
from pyneuroutils.transforms.transform import transform


class zscore(transform):
    def __init__(self, axis=-1, **kwargs):
        super().__init__()
        self.transform = self.__class__.__name__
        self.axis = axis

    def __call__(self, x):
        return stats.zscore(x,axis=self.axis)