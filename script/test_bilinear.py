"""
A script to verify that Pytorch and Tensorflow is inconsistent in bilinear upsample.
"""
import torch
import tensorflow as tf
import numpy as np

a = np.random.rand(1, 1, 5, 5).astype("float32")
x_torch = torch.from_numpy(a)
x_tf = a.transpose(0, 2, 3, 1)

x = tf.placeholder(tf.float32, [None, 5, 5, 1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

y_tf = sess.run(tf.image.resize_bilinear(x, (7, 7)), {x:x_tf})[0, :, :, 0]
y_torch = torch.nn.functional.interpolate(x_torch, (7, 7), mode='bilinear').detach().cpu().numpy()[0, 0, :, :]
print(y_tf)
print(y_torch)