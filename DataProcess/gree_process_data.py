#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : gree_process_data.py
# @Date  : 2020/9/17
# @Email  : 560331
import os

from DataProcess.vocab import get_w2i, get_gree_tag2index, unk_flag, pad_flag, cls_flag, sep_flag
from Public.path import path_gree_dir
import numpy as np


class DataProcess(object):

    def __init__(self, max_len=20, data_type='gree', model='other'):
        """
        对数据处理进行参数初始化
        :param max_len: 句子的最长长度，最长默认是20
        :param data_type:处理的文本类型
        :param model:
        """
        self.w2i = get_w2i()
        self.tag2index = get_gree_tag2index()
        #  词表的长度
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len
        self.model = model

        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)
        if data_type == 'gree':
            self.base_dir = path_gree_dir
        else:
            raise RuntimeError('type out of range must be gree')

    def get_data(self, one_hot: bool = True):

        path_train = os.path.join(self.base_dir, "train_new.txt")
        path_test = os.path.join(self.base_dir, "test_new.txt")

        train_data, train_label = self.text_to_indexs(path_train)
        test_data, test_label = self.text_to_indexs(path_test)
        # 对文本标签进行one_hot处理
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0] * self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)

            train_label = label_to_one_hot(index=train_label)

            test_label = label_to_one_hot(index=test_label)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    def text_to_indexs(self, file_path):
        data, label = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            line_data, line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    char_index = self.w2i.get(w, self.w2i[self.unk_flag])
                    tag_index = self.tag2index.get(t, 0)
                    line_data.append(char_index)
                    line_label.append(tag_index)
                else:
                    if len(line_data) < self.max_len:
                        pad_num = self.max_len - len(line_data)
                        line_data = [self.pad_index] * pad_num + line_data
                        line_label = [0] * pad_num + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    label.append(line_label)
                    line_data, line_label = [], []
        return np.array(data), np.array(label)

    def text_to_index(self, str):
        data=[]
        line_data = []
        for i, j in enumerate(str):
            char_index = self.w2i.get(j, self.w2i[self.unk_flag])
            line_data.append(char_index)
        if len(line_data) < self.max_len:
            pad_num = self.max_len - len(line_data)
            line_data = [self.pad_index] * pad_num + line_data
            data.append(line_data)
            return np.array(data)
