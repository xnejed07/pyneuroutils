from pyneuroutils.transforms.transform import transform
import numpy as np

class diff(transform):
    def __init__(self, axis, **kwargs):
        super().__init__()
        self.axis = axis

    def __call__(self,x):
        if x.ndim != 2:
            raise Exception("InvalidInputDimensins: {}".format(x.shape))
        return np.diff(x,axis=self.axis)
