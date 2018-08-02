#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ping_ft_ana.py
# @Author: He Chao
# @Date  : 18-6-6
# @Desc  : 
# @Solution  :

import pprint
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def ft_cut_anal(train_ft, ft_name, cut_num):
    range_name = ft_name + "_range"

    #　将数据进行分箱操作，cut_num表示分箱的数量，分箱就是将数据按取值范围均匀切割成几份
    train_ft[range_name] = pd.cut(train_ft[ft_name], cut_num)

    # 统计各个分箱数据下的数据量
    print(train_ft[range_name].value_counts())
    groupy_rt = train_ft[[range_name, 'label']].groupby([range_name])

    # 查看各个分箱下训练目标的label的分布是否不同
    print(groupy_rt.mean())


if __name__ == "__main__":
    train_data = pd.read_csv('./input/train_feature_demo.csv')
    ft_name = "1"
    ft_cut_anal(train_data, ft_name, cut_num=3)  #分箱数量需要根据实际情况去调

    # 输出结果
    '''
    (0.0667, 0.133]      80
    (0.133, 0.2]         11
    (-0.0002, 0.0667]     9
    Name: directiondiff_between60_90_range, dtype: int64
                                         label
    directiondiff_between60_90_range
    (-0.0002, 0.0667]                 0.000000
    (0.0667, 0.133]                   0.379953
    (0.133, 0.2]                      0.060076
    '''
    #　输出结果分析
    '''
    这里将directiondiff_between60_90这个特征的取值均匀划分成了三份
    (0.0667, 0.133]      80
    (0.133, 0.2]         11
    (-0.0002, 0.0667]     9
    从value_counts分析可以发现数据大都分布在(0.0667, 0.133]区间，数量为80个，其他两个区间数据量分别为11,9

    查看各个分箱下训练目标的label的分布是否不同
    (-0.0002, 0.0667]                 0.000000
    (0.0667, 0.133]                   0.379953
    (0.133, 0.2]                      0.060076
    从上面可以发现(0.0667, 0.133]区间的label值平均值为０，即该区间范围内80个用户的label全为０．
    (0.133, 0.2]和(-0.0002, 0.0667]的数量差不多，分别是11,9．但是两者的label均值相差很大．
    (0.0667, 0.133]区间的label均值为0.379953，(0.133, 0.2]区间的label均值为0.060076．
    0.379953远远大于0.060076，　所以(0.0667, 0.133]的驾驶风险程度大于(0.133, 0.2]．

    综上所述，　directiondiff_between60_90这个特征是个强特，能够区分无风险，低风险和高风险．
    '''
