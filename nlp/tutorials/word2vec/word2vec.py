""" Implementing word2vec using the skipgram model using a logistic regression
as probability function. """

import tensorflow as tf
import numpy as np

import math



def skipgram(vocabulary_size, embedding_size):
    """Run skip gram model for a dataset."""
    batch_size = 100 #Run sufficient batch_size until the loss is minimized

    graph = tf.Graph() #To construct a tensorflow Graph

    with graph.as_default():
        with tf.name_scope('inputs'):
            # Placeholders for inputs.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device('/cpu:0'):
            with tf.name_scode('embeddings'):
                # Get embedding
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],
                                                           -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0/math.sqrt(embedding_size)))

            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
             biases=nce_biases,
             labels=train_labels,
             inputs=embed,
             num_sampled=num_sampled,
             num_classes=vocabulary_size))

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
