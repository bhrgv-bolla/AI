from __future__ import absolute_import
from __future__ import print_function

from six.moves import urllib
import os


data_url = 'http://mattmahoney.net/dc/'

tmp_dir = os.path.dirname(os.path.realpath(__file__))  # This file location.
data_dir = os.path.join(tmp_dir, 'data')

if (not os.path.exists(data_dir)):
    os.mkdir(data_dir)


def download_data(filename):
    file_path = os.path.join(data_dir, filename)
    print('Writing file to : ', file_path)
    try:
        if(not os.path.exists(file_path)):
            urllib.request.urlretrieve(data_url + filename, file_path)
    except:
        print("exception occured")

    return file_path

def main():
    filename='text8.zip'
    data = download_data(filename)
    print('DONE DOWNLOADING: ', data)

if __name__ == '__main__':
    main()
