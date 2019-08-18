import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log",required=True,help="Location of log")
    parser.add_argument("--output_file",required=True,help="Output file")
    parser.add_argument("--title", required=True, help="Title")
    parser.add_argument("--include_first",default=False,help="Whether to include first epoch")

    flags = parser.parse_args()
    df = pd.read_csv(flags.input_log)
    if not flags.include_first:
        df = df[df.epoch != 0]
    plt.title(flags.title)
    plt.ylabel('SSD loss')
    plt.xlabel('Epoch number')
    plt.plot(df.epoch, df.loss, label='training loss')
    plt.plot(df.epoch, df.val_loss, label='val loss')
    plt.savefig(flags.output_file)

if __name__ == "__main__":
    main()