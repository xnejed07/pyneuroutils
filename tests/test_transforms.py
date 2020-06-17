import unittest
from pyneuroutils.transforms import *
import numpy as np

class test(unittest.TestCase):
    def setUp(self):
        pass

    def test_0(self):
        # single channel
        x = 10 * np.random.randn(1, 15000)

        transforms = compose([filtfilt(n=3, wn=(1 / 2500, 600 / 2500), btype='bandpass'),
                              spectrogram(fs=5000, nperseg=256, noverlap=128, nfft=1024),
                              zscore(axis=-1),
                              crop(dim_y=100)])

        y = transforms(x)
        print(transforms)
        print(y.shape)

    def test_1(self):
        # multi channel
        x = 10 * np.random.randn(10, 15000)

        transforms = compose([filtfilt(n=3, wn=(1 / 2500, 600 / 2500), btype='bandpass'),
                              spectrogram(fs=5000, nperseg=256, noverlap=128, nfft=1024),
                              zscore(axis=-1),
                              crop(dim_y=100)])

        y = transforms(x)
        print(transforms)
        print(y.shape)

    def test_2(self):
        # bipolar difference
        x = 10 * np.random.randn(2, 15000)

        transforms = compose([diff(axis=0),
                              filtfilt(n=3, wn=(1 / 2500, 600 / 2500), btype='bandpass'),
                              spectrogram(fs=5000, nperseg=256, noverlap=128, nfft=1024),
                              zscore(axis=-1),
                              crop(dim_y=100)])

        y = transforms(x)


    def test_3(self):
        # bipolar difference
        x = 10 * np.random.randn(2, 15000)

        transforms = compose([diff(axis=0),
                              filtfilt(n=3, wn=(1 / 2500, 600 / 2500), btype='bandpass'),
                              spectrogram(fs=5000, nperseg=256, noverlap=128, nfft=1024),
                              zscore(axis=-1),
                              crop(dim_y=100)])

        pkl = transforms.dumps()
        new = transforms.loads(pkl)
        y = transforms(x)
        z = new(x)
        print(new)
        self.assertTrue(np.array_equal(y,z))


    def test_4(self):
        # sample and spectrogram
        x = 10 * np.random.randn(1, 15000)
        transforms = compose([sample_1d(n=10,window=5000,seed=1),
                              spectrogram(fs=5000,nperseg=256,noverlap=128,nfft=1024)])
        y = transforms(x)
        stop = 1


    def test_5(self):
        x = 10 * np.random.randn(1, 11)
        transform = segment_1d(window=4,overlap=2)
        y = transform(x)
        self.assertSequenceEqual(y.shape,(4,4))

    def test_6(self):
        x = 10 * np.random.randn(1, 11)
        transform = segment_1d(window=4,overlap=0)
        y = transform(x)
        self.assertSequenceEqual(y.shape,(2,4))









