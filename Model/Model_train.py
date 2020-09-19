#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : Model_train.py
# @Date  : 2020/9/17
# @Email  : 560331


from sklearn.metrics import f1_score
import numpy as np

from DataProcess.gree_process_data import DataProcess
from Model.BILSTM_CRF import BILSTMCRF
import matplotlib.pyplot as plt
dp = DataProcess(data_type='gree')

train_data,train_label,test_data,test_label=dp.get_data(one_hot=True)

lstm_crf=BILSTMCRF(vocab_size=dp.vocab_size,n_class=5)

lstm_crf.creat_model()

model=lstm_crf.model

hist=model.fit(train_data,train_label,batch_size=32,epochs=5,validation_data=[test_data,test_label])

model.save('bi_crf.h5')

# 创建一个绘图窗口
# plt.figure()
# print(hist.history)
# acc=hist.history['acc']
#
# val_acc=hist.history['val_acc']
#
# loss=hist.history['loss']
#
# val_loss=hist.history['val_loss']
# epochs=range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()  # 绘制图例，默认在右上角
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()


