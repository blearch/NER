#!/usr/bin/env python
# -- coding = 'utf-8' --
# Author WangBin
# Python Version 3.6.6
# @Software:PyCharm
# @File : BILSTM_CRF.py
# @Date  : 2020/9/17
# @Email  : 560331

from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from keras_contrib.layers import CRF

from DataProcess.gree_process_data import DataProcess


class BILSTMCRF():
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate

    def creat_model(self):
        inputs = Input(shape=(None,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(x)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class)(x)
        self.crf = CRF(self.n_class, sparse_target=False)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()
        return self.model

    def compile(self):
        self.model.compile('adam',
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])