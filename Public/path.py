#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : path.py
# @Date  : 2020/9/17
# @Email  : 560331
import os

#  获取当前文件的地址
current_dir = os.path.dirname(os.path.abspath(__file__))

# 词表目录(本词表采用的是 bert预训练模型的词表)
path_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')

#  实体命名识别目录

path_gree_dir = os.path.join(current_dir, '../data/gree')

# 日志文件记录地址
path_log_dir = os.path.join(current_dir, "../log")

#  jieba的分词词典地址

path_jieba_dir = os.path.join(current_dir, '../data/dict_word/word.txt')

path_jieba_dir_func=os.path.join(current_dir,'../data/dict_word/func_word.txt')


