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



for x, t, m in dataset:
    p = model(x)
    mp.append(targets=t, probs=p,metadata=m)
mp.evaluate()