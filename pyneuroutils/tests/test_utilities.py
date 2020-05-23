import unittest

from torch.utils.data import DataLoader

from ..statistics import *
from ..tests.test_helpers import *
from ..utilities import *


class MP(ModelProgress):
    def evaluate(self, plot=False, verbose=False, threshold=None, log=True):
        results = dict()
        results['auroc'] = Statistics.auroc_score(self.probs, self.targets)
        results['auprc'] = Statistics.auprc_score(self.probs, self.targets)
        results['confusion'] = Statistics.confusion_matrix_argmax(self.probs, self.targets)
        results['f1'] = Statistics.f1_score_argmax(self.probs, self.targets)

        # if plot:
        #     k = self.probs[np.where(self.targets == 1), 1][0, :]
        #     q = self.probs[np.where(self.targets == 0), 1][0, :]
        #     plt.hist(k, alpha=0.5)
        #     plt.hist(q, alpha=0.5)
        #     plt.show()

        if verbose:
            print(results)

        if log:
            self.log(key=str(self.progress), x_dict=results)
            self.log_data()

        return results


def collate(batch):
    X = np.concatenate([k[0] for k in batch], axis=0)
    y = np.array([k[1] for k in batch])
    return X, y


class Test(unittest.TestCase):
    def setUp(self):
        self.iris = SimpleNamespace()
        self.iris.dataset = DataLoader(dataset=iris_dataset(),
                                       batch_size=10,
                                       shuffle=True,
                                       drop_last=False,
                                       collate_fn=collate)
        self.iris.classifier = iris_classifier

    def test_model_progress(self):
        mp = MP(output_directory='./test-output')
        mp.log('header', {'test0': 'Hello',
                          'test1': 'World',
                          'test2': 'This is a test',
                          'model': 'ModelName',
                          'epochs': '25'})
        # run 25 epochs

        for epoch in range(25):
            # TRAIN LOOP
            mp.newEpoch(idx=epoch, train=True)
            # run 10 iterations
            for x, t in self.iris.dataset:
                p = self.iris.classifier(x)
                mp.append(targets=t, probs=p)
            mp.evaluate(plot=True, verbose=True, threshold=None, log=True)

            # VALID LOOP
            mp.newEpoch(idx=epoch, valid=True)
            for i in range(10):
                p, t = Statistics.random_binary_classifier(100, 0.2)
                mp.append(targets=t, probs=p)
            results = mp.evaluate(plot=True, verbose=True, threshold=None, log=True)

            # TEST LOOP
            mp.newEpoch(idx=epoch, test=True)
            for i in range(10):
                p, t = Statistics.random_binary_classifier(100, 0.2)
                mp.append(targets=t, probs=p)
            results = mp.evaluate(plot=True, verbose=True, threshold=0.71, log=True)


if __name__ == '__main__':
    unittest.main()
