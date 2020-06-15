import unittest

class test(unittest.TestCase):

    def test_1(self):
        import pyneuroutils.statistics as stats
        stats.random_binary_classifier(100,0.1)
