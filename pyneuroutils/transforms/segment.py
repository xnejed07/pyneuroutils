from pyneuroutils.transforms.transform import transform
import numpy as np

class segment_1d(transform):
    def __init__(self, window,overlap, **kwargs):
        super().__init__()
        self.window = window
        self.overlap = overlap


    def __call__(self,x):
        if self.window>=x.shape[-1]:
            raise Exception("InvalidInputDimensins: {} vs window size: {}".format(x.shape,self.window))

        if x.ndim==2 and x.shape[0]==1: #shape (1,Nsamp)
            n = int(np.floor(x.shape[-1]/self.window))
            if self.overlap > 0:
                n += int(np.floor((x.shape[-1]-self.overlap)/self.window))
            y = np.zeros((n,self.window))
            idx = 0
            for i in range(n):
                y[i,:] = x[0,idx:idx+self.window]
                idx += self.window - self.overlap
        else:
            raise Exception("InvalidInputDimensins: {}".format(x.shape))

        return y
