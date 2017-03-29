import pandas as pd
import glob
from os.path import join
import os
from pprint import pprint
import deepdish.io as dio

if __name__ == '__main__':
    CROPS_DIR = '/export/home/asanakoy/workspace/OlympicSports/crops'
    OUTPUT_DIR = '/export/home/asanakoy/workspace/OlympicSports/data'

    paths = glob.glob(join(CROPS_DIR, '*'))

    category_names = list()
    for path in paths:
        if os.path.isdir(path):
            category_names.append(os.path.split(path)[1])

    print 'Categories: ({}) {}'.format(len(category_names), category_names)

    sequences = dict()
    for category_name in category_names:
        sequences[category_name] = list()
        for seq_dir in glob.glob(join(CROPS_DIR, category_name, '*')):
            if os.path.isdir(seq_dir):
                sequences[category_name].append(os.path.split(seq_dir)[1])
        sequences[category_name].sort()
    pprint(sequences)

    dio.save(join(OUTPUT_DIR, 'sequences.hdf5'), sequences)
