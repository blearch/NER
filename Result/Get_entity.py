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
from LAC import LAC

class GetEntity():
    model = load_model('../Model/bi_crf.h5', custom_objects={'CRF': CRF, 'crf_loss': crf_loss,
                                                             'crf_viterbi_accuracy': crf_viterbi_accuracy})

    def __init__(self,value):
        # 使用自己的模型去识别
        self.dp = DataProcess(data_type='gree')

        self.y_str = value
        self.y = self.dp.text_to_index(self.y_str)
        self.z = self.model.predict(self.y)
        self.num2tag = self.dp.num2tag()
        self.i2w = self.dp.i2w()
    #     开始使用百度的模型去识别数字
        self.lac=LAC(mode='lac')
    def get_entity_list(self, value):
        """
         desc:
            根据输入的内容获取相应的实体-*
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
                    char = self.i2w.get(index, ' ')
                    t_line = self.z[i]
                    t_index = np.argmax(t_line[j])
                    tag = self.num2tag.get(t_index, 'O')
                    # print(char, tag)
                    chars.append(char)
                    tags.append(tag)
        print(chars, tags)
        return chars, tags

    def get_entity(self, value):
        """
        剔除杂项，只保留有意义的标签和对应的字符
        :param value:
        :return:
        """
        chars, tags = self.get_entity_list(value)
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

    def get_device_func(self):
        """
        desc: 获取功能和设备名称的实体，并返回
        :return:
        """
        a, b, c = self.get_entity(self.y)

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
        # 使用百度模型去识别出数量
        entities=self.lac.run(self.y_str)
        result=dict(zip(entities[1],entities[0]))
        if 'm' in result.keys():
            number=[result['m']]
        else:
            number=['']
        return device, func,number




if __name__ == '__main__':
    a,b,c=GetEntity('把空调的制冷模式调到26度').get_device_func()
    print(a,b,c)
