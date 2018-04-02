""" Implementing word2vec using the skipgram model using a logistic regression
as probability function. """
import tensorflow as tf
import numpy as np
import zipfile

import math
from . import directory_util, download
import os
import collections

dataset_name = 'text8.zip'



def read_data(filename):
    """ Step 1 Read data. """
    filepath = os.path.join(directory_util.get_data_dir(), dataset_name)
    with zipfile.ZipFile(filepath) as f:
        print 'FILES IN THE ZIP: ', f.namelist()
        contents = f.read(f.namelist()[0])
        try:
            print 'SAMPLE CONTENTS OF FILE', contents[:100]
        except:
            print '!!CANNOT PRINT SAMPLE'
        data = tf.compat.as_str(contents).split() # TODO what is package compat?
        return data

download.download_data(dataset_name)
vocabulary = read_data(dataset_name)
vocabulary_size = len(vocabulary) # Taking all the words in the vocabulary!
print 'VOCABULARY SIZE: ', vocabulary_size, 'SAMPLE', vocabulary[:10]
vocabulary_size = 50000 # change so that you only take part of it.


def build_dataset(vocabulary, vocabulary_size):
    """ To count the occurences of words and mark the most common words
    as UNKOWN /UNK"""
    count = [['UNK', -1]]
    # select the top common ones.
    count.extend(collections.Counter(vocabulary).most_common(vocabulary_size - 1))
    print 'COUNT: ', count[:10]
    wordDictionary = {}
    # Build dictionary with name and count
    for word, _ in count:
        wordDictionary[word] = len(wordDictionary)
    data = []
    unknown_count = 0
    for word in vocabulary:
        index = wordDictionary.get(word, 0) # Rare words. TODO dig into this.
        if(index == 0):
            unknown_count += 1
        data.append(index)
    count[0][1] = unknown_count
    reverseWordDictionary = dict(zip(wordDictionary.values(), wordDictionary.keys()))
    return data, count, wordDictionary, reverseWordDictionary

#reverseWordDictionary => Index to Word mapping.
#wordDictionary => Name to Index mapping
#count = Top vocabulary_size-1 common words. (Unkown is one of them)
#data = The original data mapped to integers. ( If there is a word that doesn't exist in the map; the data integer would be 0)
data, count, wordDictionary, reverseWordDictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary # Don't need the vocabulary any more. Since the dictionary are there.
print 'MOST COMMON DATA: ', count[:10]
print 'SAMPLE DATA: ', data[:10], [reverseWordDictionary.get(index) for index in data[:10]]


data_index = 0 #global variable to keep track of till where the data was read last time.

def generate_batch(batch_size, window_size): #TODO improve generating a batch by using some randomness in the process.
    """
    Generate a batch for training from the dataset ( data )
    :returns
    data: The data / center words.
    labels: the target/ context words.
    """
    global data_index
    batch = np.ndarray((batch_size), dtype=np.int32)
    labels = np.ndarray((batch_size, 1), dtype=np.int32)
    current_batch_index = 0
    for batch_index in range(batch_size): #These many elements are required in both the batch and labels.
        if data_index + 2*window_size >= len(data):
            data_index = 0
        center_word_index = data_index + window_size
        context_word_indices = [index for index in range(data_index, data_index+2*window_size + 1) if index != center_word_index]
        print center_word_index, context_word_indices
        for context_word_index in context_word_indices:
            batch[current_batch_index] = data[center_word_index]
            labels[current_batch_index, 0] = data[context_word_index]
            current_batch_index += 1
            if current_batch_index >= batch_size:
                break
        if current_batch_index >= batch_size:
            break
        # Shift the window by 1
        data_index += 1
    return batch, labels


def skipgram(vocabulary_size, embedding_size):
    """Run skip gram model for a dataset."""
    batch_size = 100 #Run sufficient batch_size until the loss is minimized
    num_iterations = 1000 #loop for training.

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
             # num_sampled=num_sampled,
             num_classes=vocabulary_size))

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)


    with tf.get_default_session(graph=graph) as session:
        tf.global_variables_initializer.run()
        for batch_num in xrange(num_iterations):
            # TODO More info on num skips
            inputs, labels = generate_batch(batch_size, window_size=1, num_skips=2)

