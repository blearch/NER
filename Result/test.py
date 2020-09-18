#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : test.py
# @Date  : 2020/9/17
# @Email  : 560331

import jiagu

text='洗衣机打开两分钟'

words=jiagu.seg(text)
print(words)
ner=jiagu.ner(words)
print(ner)