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