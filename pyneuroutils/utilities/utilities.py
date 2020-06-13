import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd


class Epoch:
    def __init__(self, idx):
        self.idx = idx

        self.probs = []
        self.targets = []
        self.metadata = []
        self.state = None

    def train(self):
        self.state = 'train'

    def valid(self):
        self.state = 'valid'

    def test(self):
        self.state = 'test'

    def check(self):
        if self.state is None:
            raise Exception("INVALID STATE: state must be specified")

        if not isinstance(self.probs, list):
            raise Exception("self.probs is not a list")

        if not isinstance(self.targets, list):
            raise Exception("self.targets is not a list")

        if not isinstance(self.metadata, list):
            raise Exception("self.metadata is not a list")

    def __repr__(self):
        return "epoch:{}-{}".format(str(self.idx).zfill(3), self.state)

    def append(self, targets, probs, metadata=None):
        self.check()

        if isinstance(targets, np.ndarray):
            self.targets.append(targets)
        else:
            self.targets.append(self.tensor2numpy(targets))

        if isinstance(probs, np.ndarray):
            self.probs.append(probs)
        else:
            self.probs.append(self.tensor2numpy(probs))

        if metadata is not None:
            self.metadata.append(metadata)
        else:
            self.metadata.append(['{}'.format(str(i).zfill(6)) for i in range(targets.shape[0])])

    def tensor2numpy(self, x):
        return x.data.cpu().numpy()

    def finish(self):
        # prevent double call
        if not isinstance(self.probs, np.ndarray):
            self.probs = np.concatenate(self.probs, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
            self.metadata = np.concatenate(self.metadata, axis=0)

    def summary(self):
        if not isinstance(self.probs, np.ndarray):
            raise Exception("PROBS MUST BE NDARRAY")
        df = dict()
        df['epoch'] = self.idx
        df['state'] = self.state
        df['metadata'] = self.metadata
        df['targets'] = self.targets
        df['probs'] = [str(np.around(self.probs[i, :], 3)) for i in range(self.probs.shape[0])]

        df = pd.DataFrame(df)
        return df



class ModelProgress:
    def __init__(self, output_directory=None):
        self.progress = None
        self.path = SimpleNamespace()

        self.path.dir = output_directory
        self.create_logs()


    def create_logs(self):
        if self.path.dir is None:
            return

        self.path.dir += "/" if self.path.dir[-1] is not "/" else ""

        if not os.path.exists(self.path.dir):
            os.makedirs(self.path.dir)

        self.path.now = self.now()
        self.path.log = self.path.dir + 'log_' + self.path.now + '.txt'
        self.path.data = self.path.dir + 'data_' + self.path.now + '.txt'

        with open(self.path.log, 'w') as f:
            pass

        with open(self.path.data, 'w') as f:
            pass

    @property
    def epoch(self):
        pass

    @property
    def state(self):
        pass

    @property
    def probs(self):
        pass

    @property
    def metadata(self):
        pass

    @property
    def targets(self):
        pass

    @probs.getter
    def probs(self):
        self.progress.finish()
        return self.progress.probs

    @probs.setter
    def probs(self, value):
        raise Exception("Not accessible")

    @targets.getter
    def targets(self):
        self.progress.finish()
        return self.progress.targets

    @targets.setter
    def targets(self, value):
        raise Exception("Not accessible")

    @metadata.getter
    def metadata(self):
        self.progress.finish()
        return self.progress.metadata

    @metadata.setter
    def metadata(self, value):
        raise Exception("Not accessible")

    @epoch.getter
    def epoch(self):
        return str(self.progress.idx).zfill(3)

    @epoch.setter
    def epoch(self, value):
        raise Exception("Not accessible")

    @state.getter
    def state(self):
        return self.progress.state

    @state.setter
    def state(self, value):
        raise Exception("Not accessible")

    def now(self):
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def newEpoch(self, idx, train=False, valid=False, test=False):
        self.progress = Epoch(idx=idx)
        if train:
            self.progress.train()
        if valid:
            self.progress.valid()
        if test:
            self.progress.test()
        self.progress.check()

    def append(self, targets, probs, metadata=None):
        self.progress.append(targets, probs, metadata=metadata)

    def log(self, key, x_dict):
        x_dict['date'] = self.now()
        if self.path.dir is None:
            raise Exception("Log not specified")

        with open(self.path.log, 'a') as f:
            f.write(key + ":\n")
            for key, value in x_dict.items():
                if isinstance(value, np.ndarray):
                    value = str(np.round(value, 3).tolist())
                f.write("\t{}:{}\n".format(key, value))

    def log_data(self):
        return self.progress.summary().to_csv(self.path.data, mode='a', header=False, index=False)

    def evaluate(self):
        """
        Should be overridden
        """
        raise Exception("NotImplementedError")
