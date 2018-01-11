#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:46:19 2018

@author: evanderdcosta
"""

import tensorflow as tf

def Dense(x, units, activation, reuse=False, kernel_initializer=None,
          name='dense'):
    dense_layer = tf.layers.Dense(units=units,
                                  activation=activation,
                                  use_bias=True,
                                  kernel_initializer=kernel_initializer,
                                  _reuse=reuse,
                                  name=name)
    dense_output = dense_layer(x)
    return (dense_output, ) + tuple(dense_layer.trainable_weights)
