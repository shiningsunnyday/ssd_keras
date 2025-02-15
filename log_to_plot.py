import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--title",required=True)
parser.add_argument("--first",default=-1)
flags = parser.parse_args()
title = flags.title

flags=parser.parse_args()

summary = pd.read_csv(flags.input)
if int(flags.first) < 0:
    summary.epoch = summary.epoch + 1
else:
    summary = summary.iloc[int(flags.first):, :]

fig, ax = plt.subplots()

ax.plot(summary.epoch, summary.loss, 'b-', label='training loss')
ax.plot(summary.epoch, summary.val_loss, 'b--', label='val loss')
ax.legend()
ax.title(title)
ax.xlabel('Epoch number')
ax.ylabel('SSD Loss')
fig.savefig(flags.output)