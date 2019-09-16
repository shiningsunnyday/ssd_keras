import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--title",required=True)
parser.add_argument("--start",default=-1)
parser.add_argument("--end",default=1)

flags = parser.parse_args()
title = flags.title

flags=parser.parse_args()
lr_drops = [0, 10]
lr_label_colors = ['0.8','0.6']

summary = pd.read_csv(flags.input)
if int(flags.start) >= 0:
    summary = summary.iloc[int(flags.start):int(flags.end), :]

fig = plt.figure()
# for j in range(len(lr_drops)):
#     ax.axvline(lr_drops[j],color=lr_label_colors[len(lr_drops)-1-j])
plt.plot(summary.epoch, summary.class_loss, 'b-', label='classification train loss')
plt.plot(summary.epoch, summary.val_class_loss, 'b--', label='classification val loss')
plt.plot(summary.epoch, summary.loc_loss, 'g-', label='localization train loss')
plt.plot(summary.epoch, summary.val_loc_loss, 'g--', label='localization val loss')
plt.plot(summary.epoch, summary.loss, 'r-', label='train loss')
plt.plot(summary.epoch, summary.val_loss, 'r--', label='val loss')
plt.legend()
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.ylim(0,20)
plt.title(title)
fig.savefig(flags.output)

# python log_to_plot.py --input=../training_summaries/combined/512/allstar_pos_only.csv --output=../misc_utils/0_layers_pos_only.csv --title="0 Frozen Layers; Positive Classes Only; SGD"