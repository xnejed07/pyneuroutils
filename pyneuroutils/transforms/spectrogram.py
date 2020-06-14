import scipy.signal as sig
from pyneuroutils.transforms.transform import transform


class spectrogram(transform):
    def __init__(self, fs, nperseg, noverlap, nfft, **kwargs):
        super().__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def __call__(self, x):
        _,_,y = sig.spectrogram(x,fs=self.fs,nperseg=self.nperseg,noverlap =self.noverlap,nfft=self.nfft)
        return y
