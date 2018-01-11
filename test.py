#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:09:38 2018

@author: evanderdcosta
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from basemodel import BaseModel
from ops import Dense

class Config:
    input_dim = 784
    z_dim = 100
    
    # Generator params
    g_hidden_units = 256
    g_hidden_activation = 'relu'
    
    # Discriminator params
    d_hidden_units = 256
    d_hidden_activation = 'relu'
    
    # Model params
    save_every = 100
    batch_size = 128
    n_epochs = 10000
    
    
    
    def __init__(self):
        self.name = 'simple_gan'
        self.h_run = 1
    
    
activation_fns = {'sigmoid': tf.nn.sigmoid,
                  'softmax': tf.nn.softmax,
                  'relu': tf.nn.relu,
                  'tanh': tf.nn.tanh,
                  'linear': lambda x: x
                  }
    

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

class GAN(BaseModel):
    def __init__(self, config):
        self.input_dim, self.z_dim = config.input_dim, config.z_dim
        
        self.g_hidden_units, self.g_hidden_activation = \
            config.g_hidden_units, config.g_hidden_activation
            
        self.d_hidden_units, self.d_hidden_activation = \
            config.d_hidden_units, config.d_hidden_activation
            
        
        super(GAN, self).__init__(config)
            
            

    def build_generator(self, z):
        with tf.variable_scope('generator'):            
            g_hidden, self.g_w['w_1'], self.g_w['b_1'] = \
                Dense(z, units=self.g_hidden_units, 
                      activation=activation_fns[self.g_hidden_activation],
                      name='hidden')
                
            g_output, self.g_w['w_2'], self.g_w['b_2'] = \
                Dense(g_hidden, units=self.input_dim, 
                      activation=tf.nn.sigmoid,
                      name='output')
            self.add_summary(self.g_w)
        return g_output
                
    def build_discriminator(self, x, reuse=False):
        if(not reuse):
            with tf.variable_scope('discriminator'):
                d_hidden, self.d_w['w_1'], self.d_w['b_1'] = \
                    Dense(x, units=self.d_hidden_units, 
                          activation=activation_fns[self.d_hidden_activation],
                          reuse=reuse,
                          name='hidden')
                    
                d_output_logit, self.d_w['w_2'], self.d_w['b_2'] = \
                    Dense(d_hidden, units=1, 
                          activation=lambda x: x,
                          reuse=reuse,
                          name='output')
                self.add_summary(self.d_w)
            
        else:
            with tf.variable_scope('discriminator'):
                d_hidden = Dense(x, units=self.d_hidden_units, 
                          activation=activation_fns[self.d_hidden_activation],
                          reuse=reuse,
                          name='hidden')
                d_output_logit = Dense(d_hidden, units=1, 
                                          activation=lambda x: x,
                                          reuse=True,
                                          name='output')
        return d_output_logit
                
                
                
    def build_optimiser(self):
        with tf.variable_scope('optimiser'):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logits, 
                                                                  labels=tf.ones_like(self.d_real_logits)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, 
                                                                  labels=tf.zeros_like(self.d_fake_logits)))
            self.d_loss = d_loss_real + d_loss_fake
            
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits,
                                                                                 labels=tf.ones_like(self.d_fake_logits)))
            
            self.add_summary('d_loss', self.d_loss)
            self.add_summary('g_loss', self.g_loss)
            
            self.d_optimiser = tf.train.AdamOptimizer()
            self.g_optimiser = tf.train.AdamOptimizer()
            
            self.d_optimise_op = self.d_optimiser.minimize(self.d_loss)
            self.g_optimise_op = self.g_optimiser.minimize(self.g_loss)
            
    def build(self, sess):
        self.g_w = {}
        self.z = tf.placeholder(tf.float32, shape=(None, self.z_dim), name='z')
        self.g_output = self.build_generator(self.z)
        
        self.d_w = {}
        self.x = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='x')
        self.d_real_logits = self.build_discriminator(self.x)
        
        self.d_fake_logits = self.build_discriminator(self.g_output, reuse=True)
        
        self.build_optimiser()
        
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/{}'.format(self.model_dir),
                                            sess.graph)
        
        sess.run(tf.global_variables_initializer())
        
        
    def sample_z(self, m, n):
        return np.random.uniform(-1., 1., size=(m, n))
    
    
    def fit(self, x, sess):

        def plot_samples():
            samples = sess.run(self.g_output, {self.z: self.sample_z(16, 
                                                                self.config.z_dim)})
            fig = plot(samples)
            plt.close(fig)
        
        batch_size = self.config.batch_size
        n_epochs = self.config.n_epochs
        n_iter = int(len(x) / float(batch_size))
        
        for epoch in range(1, n_epochs+1):
            g_losses = []
            d_losses = []
            for i in tqdm(range(n_iter)):
                x_train = x[i*batch_size : (i+1)*batch_size]
                z_train = self.sample_z(x_train.shape[0], self.config.z_dim)
                d_loss, _ = sess.run([self.d_loss, self.d_optimise_op], {
                                      self.z: z_train, self.x: x_train
                                      })
                summary, g_loss, _ = sess.run([self.summary_op,
                                               self.g_loss,
                                               self.g_optimise_op], {
                                      self.z: z_train, self.x: x_train
                                      })
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                self.writer.add_summary(summary, (n_iter * epoch) + i)
            
            # End-of-epoch stuff
            print('Epoch {}'.format(epoch + 1))
                
        self.save_model(sess)
            
        
    def predict(self, x):
        raise NotImplementedError()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    tf.reset_default_graph()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    config = Config()
    config.h_run = 2
    
    model = GAN(config)
    
    with tf.Session() as sess:
        model.build(sess)
        model.fit(mnist.train.images, sess)
        samples = sess.run(model.g_output, {model.z: model.sample_z(16, config.z_dim)})