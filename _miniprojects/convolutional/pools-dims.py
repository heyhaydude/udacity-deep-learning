#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:21 2018

@author: haydude
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=4, strides=4, input_shape=(100, 100, 15)))
model.summary()