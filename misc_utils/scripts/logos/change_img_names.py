import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder",required=True,help="location for data folder")
flags = parser.parse_args()
folder_dirs = next(os.walk(flags.data_folder))[1]

def rename_dir(dir):
    for base, _, filenames in os.walk(dir):
        for file in filenames:
            if file.endswith('.jpg') and not file.startswith('img'):
                    os.rename(base + '/' + file, base + '/img' + file)

for dir in folder_dirs:
    rename_dir(flags.data_folder + '/' + dir)