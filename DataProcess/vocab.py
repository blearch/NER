#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : vocab.py
# @Date  : 2020/9/17
# @Email  : 560331

from Public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 为词表建立索引
def get_w2i(vocab_path=path_vocab):
    w2i = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


#  把数据分类建立标签 并返回
def get_gree_tag2index():
    return {
        "O": 0,
        "B-Device": 1,
        'I-Device': 2,
        'B-Func': 3,
        'I-Func': 4

    }
