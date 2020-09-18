#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : Get_entity.py
# @Date  : 2020/9/17
# @Email  : 560331

from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from DataProcess.gree_process_data import DataProcess

import numpy as np

model = load_model('../Model/bi_crf.h5', custom_objects={'CRF': CRF, 'crf_loss': crf_loss,
                                                         'crf_viterbi_accuracy': crf_viterbi_accuracy})

dp = DataProcess(data_type='gree')

y_str = '洗衣机和除湿机调到节能模式'
y = dp.text_to_index(y_str)
z = model.predict(y)
num2tag = dp.num2tag()
i2w = dp.i2w()


def get_entity_list(value):
    """
     desc:
        根据输入的内容获取相应的实体
    :param value:
        value的值为Y值
        要处理的词向量
    :return:
        返回提取的实体
    """
    chars = []
    #  char 用来保存字符
    tags = []
    #  tag 则用来保存预测的标签
    for i, x_line in enumerate(value):
        for j, index in enumerate(x_line):
            if index != 0:
                char = i2w.get(index, ' ')
                t_line = z[i]
                t_index = np.argmax(t_line[j])
                tag = num2tag.get(t_index, 'O')
                # print(char, tag)
                chars.append(char)
                tags.append(tag)

    return chars, tags


def get_entity(value):
    """
    剔除杂项，只保留有意义的标签和对应的字符
    :param value:
    :return:
    """
    chars, tags = get_entity_list(value)
    indexs = []
    new_tags = []
    new_chars = ''
    for index, i in enumerate(tags):
        if i == 'O':
            pass
        else:
            new_tags.append(i)
            new_chars += chars[index]

    for index, j in enumerate(new_tags):
        if "B-" in j:
            indexs.append(index)

    return new_tags, new_chars, indexs


def get_device_func():
    a, b, c = get_entity(y)

    device = []
    func = []
    for i in range(len(c)):
        if i + 1 <= len(c) - 1:
            if "Device" in a[c[i]]:
                # print(b[c[i]:c[i+1]])
                device.append(b[c[i]:c[i + 1]])
            else:
                func.append(b[c[i]:c[i + 1]])
        elif i == len(c) - 1:
            if "Device" in a[c[i]]:
                # print(b[c[i]:len(b)])
                device.append(b[c[i]:len(b)])
            else:
                func.append(b[c[i]:len(b)])
    return device, func

