#################################################################################
#                           USAGE PSEUDO CODE                                   #
#                                                                               #
#                                                                               #
#################################################################################
import pyneuroutils as neuro


class MP(neuro.ModelProgress):
    def evaluate(self, plot=False, verbose=False, threshold=None, log=True):
        results = dict()
        results['auroc'] = neuro.Statistics.auroc_score(self.probs, self.targets)
        results['auprc'] = neuro.Statistics.auprc_score(self.probs, self.targets)
        results['confusion'] = neuro.Statistics.confusion_matrix_argmax(self.probs, self.targets)
        results['f1'] = neuro.Statistics.f1_score_argmax(self.probs, self.targets)


        if plot:
            pass


        if verbose:
            print(results)

        if log:
            self.log(key=str(self.progress), x_dict=results)
            self.log_data()

        return results


# Create object
mp = MP(output_directory='directory_name')

# log model info
mp.log('header', {'test0': 'Hello',
                  'test1': 'World',
                  'test2': 'This is a test',
                  'model': 'ModelName',
                  'epochs': '25'})

for epoch in range(N):
    mp.newEpoch(idx=epoch,train=True)
    for x, t, m in dataset:
        p = model(x)
        mp.append(targets=t, probs=p,metadata=m)
    mp.evaluate()

    mp.newEpoch(idx=epoch,test=True)
    for x, t, m in dataset:
        p = model(x)
        mp.append(targets=t, probs=p,metadata=m)
    mp.evaluate()