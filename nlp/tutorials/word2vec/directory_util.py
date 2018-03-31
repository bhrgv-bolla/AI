import os

def get_data_dir():
    """ Creates a data dir in the current directory, if it doesn't exist """
    tmp_dir = os.path.dirname(os.path.realpath(__file__))  # This file location.
    data_dir = os.path.join(tmp_dir, 'data')
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    return data_dir