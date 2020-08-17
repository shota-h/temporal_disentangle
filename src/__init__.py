import sys
import os
import csv
import collections
import pandas as pd
import numpy as np
import shutil

def output_condition(out_dpath, *args):
    dicts = vars(args[0])
    df = pd.DataFrame.from_dict(dicts, orient='index').T
    df.to_csv('{}/conditions.csv'.format(out_dpath))


def label2onehot(label, n_class):
    onehot = np.zeros((n_class))
    onehot[label] = 1
    return onehot

def clean_directory(dpath):
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
        os.mkdir(dpath)
    else:
        os.makedirs(dpath)
        
class SetIO():
    """with構文でI/Oを切り替えるためのクラス"""
    def __init__(self, filename: str):
        self.filename = filename

    def __enter__(self):
        sys.stdout = _STDLogger(out_file=self.filename)

    def __exit__(self, *args):
        sys.stdout = sys.__stdout__

class _STDLogger():
    """カスタムI/O"""
    def __init__(self, out_file='out.log'):
        self.log = open(out_file, "a+")

    def write(self, message):
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        pass
