# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:46:05 2017

@author: link9
"""

import tensorflow as tf
import numpy as np

class GRU_Cell(object):

    """
        GRU
    """

    def __init__(self, input_size, hidden_layer_size, target_size_code, target_size_time):

        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size_code = target_size_code
        self.target_size_time = target_size_time

        # Weights for input and hidden tensor
        self.Wx = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size], stddev=0.04))
        self.Wr = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size], stddev=0.04))
        self.Wz = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size], stddev=0.04))

        self.br = tf.Variable(tf.truncated_normal([self.hidden_layer_size], stddev=0.04))
        self.bz = tf.Variable(tf.truncated_normal([self.hidden_layer_size], stddev=0.04))

        self.Wh = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.hidden_layer_size], stddev=0.04))

        # Weights for output layer
        self.Wc = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size_code], stddev=0.04))
        self.Wt = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size_time], stddev=0.04))

        self.bc = tf.Variable(tf.truncated_normal([self.target_size_code], stddev=0.04))
        self.bt = tf.Variable(tf.truncated_normal([self.target_size_time], stddev=0.04))
        
        self.initial_hidden = tf.placeholder(tf.float32,
                                      shape=[None, hidden_layer_size],
                                      name='initial_hidden')
 
      
    # Function for GRU cell
    def GRU(self, previous_hidden_state, x):
        """
        GRU Equations
        """
        z = tf.sigmoid(tf.matmul(x, self.Wz) + self.bz) # [batch_size, hidden_size]
        r = tf.sigmoid(tf.matmul(x, self.Wr) + self.br)
   
        h_ = tf.tanh(tf.matmul(x, self.Wx) +
                     tf.matmul(previous_hidden_state, self.Wh) * r)
        current_hidden_state = tf.multiply(
            (1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state # [batch_size, hidden_size]

    # Function for getting all hidden state.
    def get_states(self, batch_input):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.GRU,
                                    batch_input,
                                    initializer=self.initial_hidden,
                                    name='states')

        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output_code(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.matmul(hidden_state, self.Wc) + self.bc

        return output

    # Function to get output from a hidden layer
    def get_output_time(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wt) + self.bt)
        # sig used for mnist data, relu for original paper

        return output
        

    # Function for getting all output layers
    def get_outputs_code(self, batch_input):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states(batch_input)
 
        all_outputs = tf.map_fn(self.get_output_code, all_hidden_states) # [timestep, batch_size, target_size]

        return all_outputs
        
        # Function for getting all output layers
    def get_outputs_time(self, batch_input):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states(batch_input)
        
        all_outputs = tf.map_fn(self.get_output_time, all_hidden_states) # [timestep, batch_size, target_size]
        
        return all_outputs
        