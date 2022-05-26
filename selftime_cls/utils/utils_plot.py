# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

def show_samples(X_train, y_train, dataset_name, figname='', num_shown=5):
    '''

    :param X_train:
    :param y_train:
    :param shown_num:
    :return:
    '''
    num_cls = np.max(y_train)+1

    samples={}
    for cls in range(num_cls):
        idx = np.where(y_train==cls)[0]
        # np.random.shuffle(idx)
        samples[cls] = X_train[idx[:num_shown]]

    plt.figure(figsize=(num_shown*3, num_cls))
    for i in range(1, num_cls+1):
        for j in range(1, num_shown+1):
            plt.subplot(num_cls, num_shown, j+(i-1)*num_shown)
            plt.plot(samples[i-1][j-1])
    plt.tight_layout()

    if not os.path.exists('Samples'):
        os.makedirs('Samples')
    plt.savefig('Samples/{}_{}.png'.format(dataset_name, figname))
    plt.close()


