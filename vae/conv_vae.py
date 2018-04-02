import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
im_dim = 28
c = 0
lr = 1e-3


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

# =============================== Q(z|X) ======================================
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

def Q(X):
    with tf.variable_scope('Q', reuse=tf.AUTO_REUSE):
        X_reshape = tf.reshape(X, [-1,im_dim,im_dim,1])
        conv_1 = conv2d(X_reshape, 1, [3,3])
        conv_reshape = tf.reshape(conv_1, [-1,im_dim*im_dim])
        h = fully_connected(conv_reshape, h_dim, activation_fn=tf.nn.relu)
        z_mu = fully_connected(h, z_dim, activation_fn=None)
        z_logvar = fully_connected(h, z_dim, activation_fn=None)
        return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================
def P(z):
    with tf.variable_scope('P', reuse=tf.AUTO_REUSE):
        z_reshape = tf.reshape(z, [-1,10,10,1])
        deconv_1 = conv2d_transpose(z_reshape, 1, [3,3], stride=2)
        deconv_reshape = tf.reshape(deconv_1, [-1,400])
        h = fully_connected(deconv_reshape, h_dim, activation_fn=tf.nn.relu)
        logits = fully_connected(h, X_dim, activation_fn=None)
        prob = tf.nn.sigmoid(logits)
        return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(20000):
    X_mb, _ = mnist.train.next_batch(mb_size)
    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)