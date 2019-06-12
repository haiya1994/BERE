import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plotPR(config):
    result_dir = os.path.join(config.RESULT_DIR, config.DATA_SET)

    recall = np.load(os.path.join(result_dir, "BERE_x.npy"))
    precision = np.load(os.path.join(result_dir, "BERE_y.npy"))
    auc = metrics.auc(x=recall, y=precision)

    print('Area under the curve: {:.4}'.format(auc))

    plt.plot(recall[:], precision[:], label='BERE' + ': {0:0.3f}'.format(auc), color='red', lw=1, marker='o',
             markevery=0.1, ms=6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    base_list = ['BiGRU+2ATT', 'BiGRU+ATT', 'PCNN+ATT', 'PCNN+AVE']
    color = ['purple', 'darkorange', 'green', 'xkcd:azure']
    marker = ['d', 's', '^', '*']

    for i, baseline in enumerate(base_list):
        recall = np.load(os.path.join(result_dir, baseline + '_x.npy'))
        precision = np.load(os.path.join(result_dir, baseline + '_y.npy'))
        auc = metrics.auc(x=recall, y=precision)
        print("\n[{}] auc: {}".format(baseline, auc))
        # plt.plot(recall, precision, color=color[i], label=baseline, lw=1, marker=marker[i], markevery=0.1, ms=6)
        plt.plot(recall, precision, label=baseline + ': {0:0.3f}'.format(auc), color=color[i], lw=1, marker=marker[i],
                 markevery=0.1, ms=6)

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc="upper right", prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_path = os.path.join(result_dir, "pr.pdf")
    plt.savefig(plot_path)
    print('Precision-Recall plot saved at: {}'.format(plot_path))


if __name__ == '__main__':
    from data.dti import config

    plotPR(config)
