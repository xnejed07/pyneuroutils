import scipy.signal as sig
from pyneuroutils.transforms.transform import transform

class filtfilt(transform):
    def __init__(self, n, wn, btype='low', **kwargs):
        super().__init__()
        self.n = n
        self.wn = wn
        self.btype = btype
        self._b,self._a = sig.butter(n,wn,btype)

    def __call__(self,x):
        if x.ndim != 2:
            raise Exception("InvalidInputDimensins: {}".format(x.shape))
        return sig.filtfilt(self._b,self._a,x)

