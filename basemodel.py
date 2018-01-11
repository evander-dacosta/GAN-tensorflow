#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:11:14 2018

@author: evanderdcosta
"""

import tensorflow as tf
import os
import gym
import inspect

def class_vars(obj):
    """
    Collects all hyperparameters from the config file
    we know a hyperparam b/c it's preceded by a 'h_'
    e.g. h_optimizer
    """
    return{k:v for k,v in inspect.getmembers(obj) 
            if k.startswith('h_') and not callable(k)}


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self._saver = None
        
        self._attrs = class_vars(config)
        print(self._attrs)
        
        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))
    
        self._example_state = None
        

    def _add_summary(self, name, var):
        ndims = var.shape.ndims
        with tf.name_scope('{}_summaries'.format(name)):
            if(ndims >= 1):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('{}_mean'.format(name), mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('{}_stddev'.format(name), stddev)
                tf.summary.scalar('{}_max'.format(name), tf.reduce_max(var))
                tf.summary.scalar('{}_min'.format(name), tf.reduce_min(var))
                tf.summary.histogram('{}_histogram'.format(name), var)
            else:
                tf.summary.scalar(name, var)
                
    def add_summary(self, *args):
        if(len(args) == 1):
            var_dict = args[0]
            if(not isinstance(var_dict, dict)):
                raise ValueError("Argument given to Model.add_summary must be a 'dict' type"
                                 "given {}".format(type(var_dict)))
            summary_variables = var_dict.items()
            for name, var in summary_variables:
                self._add_summary(name, var)
                
        elif(len(args) == 2):
            name, var = args
            self._add_summary(name, var)
            
        else:
            raise Exception("Unknown number of arguments."
                            "Needs a name and a Tensorflow variable")
            
        
    def save_model(self, sess, step=None):
        print("[*] Saving a checkpoint")
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(sess, self.checkpoint_dir)
        
    def load_model(self):
        print("[*] Loading a model")
        chkpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if(chkpt and chkpt.model_checkpoint_path):
            chkpt_name = os.path.basename(chkpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, chkpt_name)
            self.saver.restore(self.sess, fname)
            print("[*] SUCCESS!")
            return True
        else:
            print("Model load failed....")
            return False
        
    @property
    def action_size(self):
        return self._env.action_space.n
    
    @property
    def state_shape(self):
        if(self._example_state is None):
            self._example_state = self._env.reset()
        return len(self._example_state)
        
    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)
    
    @property
    def model_dir(self):
        model_dir = self.config.name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                         if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if(self._saver == None):
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
    
    def build(self):
        raise NotImplementedError()
        
        
    def fit(self, x, y):
        raise NotImplementedError()
        
    def predict(self, x):
        raise NotImplementedError()