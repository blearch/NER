#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : test.py
# @Date  : 2020/9/19
# @Email  : 560331


from keras import Sequential

from Model.crf import CRF

model=Sequential()
model.add(tf.keras.layers.Input(shape=(maxLen,)))
model.add(tf.keras.layers.Embedding(vocabSize, 100))
crf=CRF(5,name='crf_layer')
model.add(crf)
model.compile('adam',loss={'crf_layer': crf.get_loss})
model.fit(X, np.argmax(y,axis=-1))