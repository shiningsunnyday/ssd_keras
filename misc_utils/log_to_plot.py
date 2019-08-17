import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--title",required=True)
parser.add_argument("--first",default=-1)
parser.add_argument("--last",default=1)

flags = parser.parse_args()
title = flags.title

flags=parser.parse_args()
lr_drops = [0, 10, 30, 100]
lr_label_colors = ['0.8','0.6','0.4','0.2']

summary = pd.read_csv(flags.input)
if int(flags.first) >= 0:
    summary = summary.iloc[int(flags.first):int(flags.last), :]

fig = plt.figure()
ax = fig.add_subplot()
for j in range(len(lr_drops)):
    ax.axvline(lr_drops[j],color=lr_label_colors[len(lr_drops)-1-j])
ax.plot(summary.epoch, summary.loss, 'b-', label='training loss')
ax.plot(summary.epoch, summary.val_loss, 'b--', label='val loss')
ax.legend()
ax.set(xlabel='Epoch number',ylabel='SSD Loss',title=title)

fig.savefig(flags.output)

# python log_to_plot.py --input=../training_summaries/logos_relabelled/0_layers_pos_only.csv --output=../misc_utils/0_layers_pos_only.csv --title="0 Frozen Layers; Positive Classes Only; SGD"