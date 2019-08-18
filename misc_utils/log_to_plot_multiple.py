import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_logs",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--title",required=True)
parser.add_argument("--first",default=-1)

flags = parser.parse_args()
title = flags.title

summaries = flags.input_logs.split(',')
labels = ['0 frozen layers','3 frozen layers', '6 frozen layers', '10 frozen layers', '14 frozen layers']
lr_drops = [[1,6,35,64],[1,8,36,64],[1,6,36,64],[1,10,42,67],[1,11,37,82]] # (num labels, num drops)
lr_label_colors = ['0.8','0.6','0.4','0.2']
heights = [11,9,7,5]
lr_label_text = ['0.0005','0.0001','0.00005','0.00001']
colors = 'bgrcm'

fig = plt.figure(figsize=(20,4))
for i in range(len(summaries)):
    log = summaries[i]
    summary = pd.read_csv(log)
    if int(flags.first) < 0:
        summary.epoch = summary.epoch + 1
    else:
        summary = summary.iloc[int(flags.first):, :]
    ax = fig.add_subplot(1,len(summaries),i+1)
    for j in range(len(lr_drops[i])):
        ax.axvline(lr_drops[i][j],color=lr_label_colors[len(lr_drops[i])-1-j])
        # ax.text(lr_drops[i][j],heights[j],lr_label_text[j],fontsize=8)
    ax.set_ylim(2,12)
    ax.plot(summary.epoch, summary.loss, colors[i]+'-', label='train loss')
    ax.plot(summary.epoch, summary.val_loss, colors[i]+'--', label='val loss')
    ax.legend(fontsize='xx-small')
    ax.set_title(labels[i],fontdict={'fontsize':10})
    ax.set(xlabel='Epoch number',ylabel='SSD Loss')
fig.savefig(flags.output)

# fig = plt.figure()
# ax = fig.subplots()
# for i in range(len(summaries)):
#     log = summaries[i]
#     summary = pd.read_csv(log)
#     if int(flags.first) < 0:
#         summary.epoch = summary.epoch + 1
#     else:
#         summary = summary.iloc[int(flags.first):, :]
#     for j in range(len(lr_drops[i])):
#         ax.axvline(lr_drops[i][j],color=lr_label_colors[len(lr_drops[i])-1-j])
#         ax.text(lr_drops[i][j],10,lr_label_text[j],fontsize=12)
#     ax.plot(summary.epoch, summary.loss, colors[i]+'-', label=labels[i]+' training loss')
#     ax.plot(summary.epoch, summary.val_loss, colors[i]+'--', label=labels[i]+' val loss')
#     ax.legend(fontsize='xx-small')
#     ax.set(xlabel='Epoch number',ylabel='SSD Loss',title=title)
# fig.savefig(flags.output)

# SAMPLE CODE COMMENT OUT

# python misc_utils/log_to_plot_multiple.py --input_logs="training_summaries/belgas_relabelled_frozen_layers_do_over/0_layers_sgd_redo.csv,training_summaries/belgas_relabelled_frozen_layers_do_over/3_layers_sgd_redo.csv,training_summaries/belgas_relabelled_frozen_layers_do_over/6_layers_sgd_redo.csv,training_summaries/belgas_relabelled_frozen_layers_do_over/10_layers_sgd_redo.csv,training_summaries/belgas_relabelled_frozen_layers_do_over/14_layers_sgd_redo.csv" --output=misc_utils/all_layers_redo.png --title="Frozen layer experiment; SGD; manual learning rate drops"