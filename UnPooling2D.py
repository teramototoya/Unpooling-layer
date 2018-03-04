import numpy as np
import cv2
import sys
import os
import datetime
import matplotlib.pyplot as plt


import tensorflow as tf

from keras.datasets import cifar10
from keras.utils import np_utils, conv_utils
from keras.models import load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D ,UpSampling2D, Input, Conv2DTranspose
from keras.engine import Layer, InputSpec
from keras.layers.pooling import _Pooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras import backend as K

import math
import scipy.misc
import cv2

import math
config =tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",allow_growth=True))
tf.Session(config=config)





class MaxPooling2D_with_argmax(_Pooling2D):
    def __init__(self,
                 pool_size   = (2, 2),
                 strides     = None,
                 padding     = 'valid',
                 data_format = None, **kwargs):
        super(MaxPooling2D_with_argmax, self).__init__(pool_size, strides, padding, data_format, **kwargs)

    def call(self, inputs):
        padding   = self._preprocess_padding(self.padding)
        strides   = (1,) + self.strides + (1,)
        pool_size = (1,) + self.pool_size + (1,)
        output, argmax = self._pooling_function(inputs      = inputs,
                                                pool_size   = pool_size,
                                                strides     = strides,
                                                padding     = padding,
                                                data_format = self.data_format)
        self.argmax = argmax
        return output


    def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
        output, argmax = tf.nn.max_pool_with_argmax(inputs,
                                                    ksize=pool_size,
                                                    strides=strides,
                                                    padding=padding)

        outputs = [output, argmax]
        self.n_output = len(outputs)
        return outputs


    def _preprocess_padding(self, padding):
        """Convert keras' padding to tensorflow's padding.
        # Arguments
            padding: string, `"same"` or `"valid"`.
        # Returns
            a string, `"SAME"` or `"VALID"`.
        # Raises
            ValueError: if `padding` is invalid.
        """
        if padding == 'same':
            padding = 'SAME'
        elif padding == 'valid':
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding:', padding)
        return padding


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])



    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format,
                  'argmax': self.argmax}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UnPooling2D(Layer):
    def __init__(self,
                 Rectified_FM,
                 pool_size    = (2, 2),
                 strides      = (1, 1),
                 padding      = 'valid',
                 data_format  = 'channels_last',
                 indices      = None, **kwargs):
        super(UnPooling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.pool_size   = conv_utils.normalize_tuple(pool_size, 2, 'size')
        self.input_spec  = InputSpec(ndim=4)
        self.indices     = indices
        self.strides     = strides
        self.padding     = padding
        self.Rectified_FM  = Rectified_FM
        self.unpooling_output_shape = Rectified_FM.get_shape().as_list()



    def compute_output_shape(self, input_shape):
        if self.unpooling_output_shape is None:
            output_shape = list(input_shape)
            if self.data_format == 'channels_first':
                c_axis, h_axis, w_axis = 1, 2, 3
            else:
                c_axis, h_axis, w_axis = 3, 1, 2

            kernel_h, kernel_w = self.pool_size
            stride_h, stride_w = self.strides

            output_shape[c_axis] = input_shape[c_axis]
            output_shape[h_axis] = conv_utils.deconv_length(
                output_shape[h_axis], stride_h, kernel_h, self.padding)
            output_shape[w_axis] = conv_utils.deconv_length(
                output_shape[w_axis], stride_w, kernel_w, self.padding)
        else:
            output_shape = self.unpooling_output_shape
        return tuple(output_shape)

    def Unpool(self,pooled_Maps, indices,
               Rectified_FM = None,
               pool_size    = [1,2,2,1],
               strides      = [1,1,1,1],
               padding      = 'valid',
               scope        = 'unpool'):
        """
           Unpooling layer after max_pool_with_argmax.
           The name of args cited from this paper(Fig.1 at https://arxiv.org/pdf/1311.2901.pdf)
           Args:
               pooled_Maps  : max pooled output tensor
               indices      : argmax indices
               Rectified_FM : Rectified Feature Maps

           Return:
               unpool:   unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape = tf.shape(pooled_Maps)
            if Rectified_FM is None:
                # Calculate output shape
                height, width, filters = input_shape[1:]
                kernel_h, kernel_w = pool_size[1:3]
                stride_h, stride_w = strides[1:3]

                out_height = conv_utils.deconv_length(height, stride_h, kernel_h, padding)
                out_width  = conv_utils.deconv_length(width,  stride_h, kernel_h, padding)

                output_shape = [input_shape[0], out_height, out_width, filters]
            else:
                output_shape = Rectified_FM.get_shape().as_list()
                if output_shape[0] is None:
                    output_shape = [input_shape[0], ] + output_shape[1:]

            flat_input_size   = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            flat_pooled = tf.reshape(pooled_Maps, tf.stack([flat_input_size]))

            batch_range = tf.reshape(tensor = tf.range(tf.cast(input_shape[0], tf.int64),
                                     dtype  = indices.dtype),
                                     shape  = [input_shape[0], 1, 1, 1])
            b = tf.ones_like(indices) * batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            flat_indices = tf.reshape(indices, tf.stack([flat_input_size, 1]))
            flat_indices = tf.concat([b, flat_indices], 1)

            #Switches is Fig.1 at https://arxiv.org/pdf/1311.2901.pdf
            Switches = tf.scatter_nd(flat_indices, tf.ones_like(flat_pooled), shape=tf.cast(flat_output_shape, tf.int64))
            Switches = tf.reshape(Switches, [-1])
            Switches = tf.greater(Switches, tf.zeros_like(Switches))
            Switches = tf.reshape(Switches, tf.stack(output_shape))

            img = tf.image.resize_nearest_neighbor(pooled_Maps, output_shape[1:3])
            Unpooled_Maps = tf.multiply(img, tf.cast(Switches, img.dtype))

            return Unpooled_Maps

    def call(self, inputs):
        return self.Unpool(inputs, self.indices, Rectified_FM = self.Rectified_FM)

    def get_config(self):
        config = {'size': self.pool_size,
                  'data_format': self.data_format}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
