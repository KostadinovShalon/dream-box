import json

import numpy as np
import sklearn.metrics as sk
import argparse

parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
parser.add_argument('id_results', help='id results file')
parser.add_argument('ood_results', help='ood results file')
parser.add_argument('--det-score-threshold', type=float, default=0.5)
args = parser.parse_args()

id_data = json.load(open(args.id_results, 'r'))
ood_data = json.load(open(args.ood_results, 'r'))

det_score_th = args.det_score_threshold

id_data = [d for d in id_data if d['score'] > det_score_th]
ood_data = [d for d in ood_data if d['score'] > det_score_th]

id_ood_scores = [d['ood_score'] for d in id_data]
ood_ood_scores = [d['ood_score'] for d in ood_data]

id_ood_scores.sort()
th = id_ood_scores[int(0.95 * len(id_ood_scores))]

print('Threshold: ', th)
print('FPR95: ', sum([1 for s in ood_ood_scores if s < th]) / len(ood_ood_scores))

# Plot PDFs
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(id_ood_scores, label='ID')
sns.kdeplot(ood_ood_scores, label='OOD')
plt.legend()
plt.show()


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95,
                          pos_label=None, return_index=False):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # import ipdb;
    # ipdb.set_trace()
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]
    recall_fps = fps / fps[-1]
    # breakpoint()
    ## additional code for calculating.
    if return_index:
        recall_level_fps = 1 - recall_level
        index_for_tps = threshold_idxs[np.argmin(np.abs(recall - recall_level))]
        index_for_fps = threshold_idxs[np.argmin(np.abs(recall_fps - recall_level_fps))]
        index_for_id_initial = []
        index_for_ood_initial = []
        for index in range(index_for_fps, index_for_tps + 1):
            if y_true[index] == 1:
                index_for_id_initial.append(desc_score_indices[index])
            else:
                index_for_ood_initial.append(desc_score_indices[index])
    # import ipdb;
    # ipdb.set_trace()
    ##
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    # 8.868, ours
    # 5.772, vanilla
    # 5.478, vanilla 18000
    # 6.018, oe
    # 102707,
    # 632
    # 5992
    # breakpoint()
    if return_index:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), index_for_id_initial, index_for_ood_initial
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))
    # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95, return_index=False, plot=False):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    if plot:
        # breakpoint()
        import matplotlib.pyplot as plt
        fpr1, tpr1, thresholds = sk.roc_curve(labels, examples, pos_label=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr1, tpr1, linewidth=2,
                label='10000_1')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.legend(fontsize=12)
        plt.savefig('10000_1.jpg', dpi=250)
    aupr = sk.average_precision_score(labels, examples)
    if return_index:
        fpr, index_id, index_ood = fpr_and_fdr_at_recall(labels, examples, recall_level, return_index=return_index)
        return auroc, aupr, fpr, index_id, index_ood
    else:
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
        return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


measures = get_measures(-np.array(id_ood_scores), -np.array(ood_ood_scores), plot=True)
print_measures(measures[0], measures[1], measures[2], 'energy')
