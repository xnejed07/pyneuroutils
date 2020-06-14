from pyneuroutils.transforms.transform import transform
import numpy as np

class sample_1d(transform):
    def __init__(self, n,window,seed=None, **kwargs):
        super().__init__()
        self.n = n
        self.seed = seed
        self.window = window


    def __call__(self,x):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.window>=x.shape[-1]:
            raise Exception("InvalidInputDimensins: {} vs window size: {}".format(x.shape,self.window))

        if x.ndim==2 and x.shape[0]==1: #shape (1,Nsamp)
            y = np.zeros((self.n,self.window))
            for i in range(self.n):
                idx = np.random.randint(x.shape[-1]-self.window-1)
                y[i,:] = x[0,idx:idx+self.window]
        else:
            raise Exception("InvalidInputDimensins: {}".format(x.shape))

        return y
