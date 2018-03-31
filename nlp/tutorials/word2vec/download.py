from __future__ import absolute_import
from __future__ import print_function

from six.moves import urllib
import os
from . import directory_util

data_url = 'http://mattmahoney.net/dc/'

data_dir = directory_util.get_data_dir()


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
