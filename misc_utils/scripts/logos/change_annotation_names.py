import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--text_file",required=True,help="location for images")
parser.add_argument("--out_file",required=True,help="location for output")

flags = parser.parse_args()
text = np.loadtxt(flags.text_file,dtype=str)

# for i in range(len(text)):
#     ind = text[i].index('img')
#     text[i] = str(text[i][:ind]) + '_' + str(text[i][ind:])

# for i in range(len(text)):
#     ind = text[i].index('_')
#     text[i] = text[i][:ind] + '/' + text[i][ind+1:]

for i in range(len(text)):
    ind = text[i].index('img')
    text[i] = text[i][:ind] + '/' + text[i][ind:]

np.savetxt(flags.out_file, text, fmt="%s")