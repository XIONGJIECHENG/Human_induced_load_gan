# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:24:49 2018

@author: DELL
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio  


X_dim = 2000
z_dim = 100
y_dim = 12
h1_dim = 1000
h2_dim = 500
h3_dim = 250
h4_dim = 125


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim+y_dim, h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h1_dim]))

D_W2 = tf.Variable(xavier_init([h1_dim+y_dim, h2_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))

D_W3 = tf.Variable(xavier_init([h2_dim+y_dim, h3_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[h3_dim]))

D_W4 = tf.Variable(xavier_init([h3_dim+y_dim, h4_dim]))
D_b4 = tf.Variable(tf.zeros(shape=[h4_dim]))

D_W5 = tf.Variable(xavier_init([h4_dim, 1]))
D_b5 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2, D_W3, D_b3, D_W4, D_b4, D_W5, D_b5]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim+y_dim, h4_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h4_dim]))

G_W2 = tf.Variable(xavier_init([h4_dim+y_dim, h3_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h3_dim]))

G_W3 = tf.Variable(xavier_init([h3_dim+y_dim, h2_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[h2_dim]))

G_W4 = tf.Variable(xavier_init([h2_dim+y_dim, h1_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[h1_dim]))

G_W5 = tf.Variable(xavier_init([h1_dim, X_dim]))
G_b5 = tf.Variable(tf.zeros(shape=[X_dim]))

G_W6 = tf.Variable(tf.truncated_normal([10, 1, 1]))



theta_G = [G_W1, G_W2, G_b1, G_b2, G_W3, G_b3, G_W4, G_b4, G_W5, G_b5, G_W6]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h1 = tf.concat(axis=1, values=[G_h1, y])
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h2 = tf.concat(axis=1, values=[G_h2, y])
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h3 = tf.concat(axis=1, values=[G_h3, y])
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_h5 = tf.nn.relu(tf.matmul(G_h4, G_W5) + G_b5)
    G_h6 = tf.reshape(G_h5, [-1, 1, 2000])
    G_h7 = tf.nn.conv1d(G_h6, G_W6, stride=1, padding='SAME', data_format="NCW")
    G_h8 = tf.nn.relu(tf.reshape(G_h7, [-1, 2000]))
    return G_h8


def D(X, y):
    inputs = tf.concat(axis=1, values=[X, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h1 = tf.concat(axis=1, values=[D_h1, y])
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h2 = tf.concat(axis=1, values=[D_h2, y])
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_h3 = tf.concat(axis=1, values=[D_h3, y])
    D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
    out = tf.matmul(D_h4, D_W5) + D_b5
    return out


G_sample = G(z, y)
D_real = D(X, y)
D_fake = D(G_sample, y)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, 'E:\\GAN_human_load\\check_point_0330\\my_model-1000000')

y_sample = np.zeros(shape=[1000, y_dim])

for i in range(12):
    y_sample[:,i]=1
    samples = sess.run(G_sample, feed_dict={z: sample_z(1000, z_dim), y:y_sample})
    y_sample = np.zeros(shape=[1000, y_dim])
    sio.savemat('E:\\GAN_human_load\\函数及汇总数据\\gene_samples{}.mat'.format(str(i)), {'Gene_load': samples}) 