#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gini_coefficient.py
# @Author: xxx
# @Date  : 18-4-10
# @Desc  : 
# @Solution  :

import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


if __name__ == "__main__":
    # 用户id的顺序
    actual = [99, 1, 22, 45, 5, 6, 7, 100]
    # 完全相逆的预测结果, gini: 0.647
    predictions = [100, 7, 6, 5, 45, 22, 1, 99]
#    # 后半部分相逆的预测结果, gini: 0.824
#    predictions = [99, 1, 22, 100, 7, 6, 5, 45]
#    # 变换后两个位置的预测结果, gini: 0.425
#    predictions = [99, 1, 22, 45, 5, 6, 100, 7]
#    # id降序排列的预测结果, gini: 0.039
#    predictions = [100, 99, 45, 22, 7, 6, 5, 1]
#    # id升序排列的预测结果, gini: -0.039
#    predictions = [1, 5, 6, 7, 22, 45, 99, 100]
#    # id前后半部对调的预测结果, gini: 0.066
#    predictions = [5, 6, 7, 100, 99, 1, 22, 45]
#    # 完全相同id顺序的预测结果, gini: 1
    predictions = [99, 1, 22, 45, 5, 6, 7, 100]

    gini_predictions = gini(actual, predictions)
    gini_max = gini(actual, actual)
    ngini= gini_normalized(actual, predictions)
    print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))


    