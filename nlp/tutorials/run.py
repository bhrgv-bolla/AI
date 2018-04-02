""" To run word2vec """

from word2vec import word2vec as wv



print wv.generate_batch(batch_size=20, window_size=1, num_skips=2)