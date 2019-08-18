import urllib.request
import argparse
import numpy as np
import os
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder",required=True,help="location for data folder")
flags = parser.parse_args()
folder_dirs = next(os.walk(flags.data_folder))[1]

def fetch_from_dir(dir):
    if dir == '0samples':
        return
    print("Directory", dir)
    dir = flags.data_folder + '/' + dir
    urls_txt = np.loadtxt(dir + '/urls.txt',dtype=str)
    for url in urls_txt:
        try:
            urllib.request.urlretrieve(str(url[1]).replace('http', 'https'), dir + '/' + str(url[0]) + '.jpg')
        except Exception as e:
            print("https failed for", str(url[0]) + '.jpg', 'because', e)
        try:
            urllib.request.urlretrieve(str(url[1]), dir + '/' + str(url[0]) + '.jpg')
        except Exception as e:
            print("http failed for", str(url[0]) + '.jpg', 'because', e)
    print("Directory", dir, "complete")

dirs = folder_dirs
dirs = dirs[dirs.index('BMW'):]

pool = Pool(processes=24)
pool.map(fetch_from_dir, dirs)