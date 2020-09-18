#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : gree_data_processing.py
# @Date  : 2020/9/17
# @Email  : 560331

"""

desc:
    主要用于对数据文件的切割处理
"""

import os
from Public.path import path_gree_dir

def gree_data_processing(split_rate: float=0.8,ignore_exist: bool=False)->None:
    """
       用于处理gree的标注数据
       :param split_rate: 训练集和测试集的切分比例
       :param ignore_exist:    是否忽略已经存在的文件，（忽略之后就不会处理第二次）
       :return: None
       """
    path = os.path.join(path_gree_dir, 'all_new.txt')
    path_train = os.path.join(path_gree_dir, 'train_new.txt')
    path_test = os.path.join(path_gree_dir, 'test_new.txt')

    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        return
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        line_t = []
        for l in f:
            if l != '\n':
                line_t.append(l)
            else:
                texts.append(line_t)
                line_t = []

    if split_rate >= 1.0:
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # 分割和存数文本
    def split_save(texts: [str], save_path: str) -> None:
        data = []
        for line in texts:
            for item in line:
                data.append(item)
            data.append("\n")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("".join(data))

    split_save(texts=train_texts, save_path=path_train)
    split_save(texts=test_texts, save_path=path_test)