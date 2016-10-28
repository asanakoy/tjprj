import h5py
import numpy as np


def get_num_classes(indices_path):
    mat_data = h5py.File(indices_path, 'r')
    num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    return num_cliques
