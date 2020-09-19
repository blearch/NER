#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : test.py
# @Date  : 2020/9/17
# @Email  : 560331

from LAC import LAC
import time
lac=LAC(mode='lac')
a=time.time()
text=u"空调打开半小时"

lac_result=lac.run(text)
b=time.time()-a
print(dict(zip(lac_result[1],lac_result[0])))
print(lac_result)
print('使用了时间')
print(b)