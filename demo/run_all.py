import sys
import numpy as np
# from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

import tfeval.olympicsports.compute_sim
# import train
# import train_cliquecnn_reassign as train
import convnet.train as train


def train_and_compute_sim(gpu_id_str, category):
    train.main([gpu_id_str, category])
    tfeval.olympicsports.compute_sim.main([gpu_id_str, category])


if __name__ == '__main__':
    gpu_id_str = 0
    chunk_ids = [int(sys.argv[1])]
    num_chunks = int(sys.argv[2])
    if len(sys.argv) > 3:
        chunk_ids.append(int(sys.argv[3]))

    print 'Running worker on GPU:{} on chunks {} from {}'.format(gpu_id_str, chunk_ids, range(num_chunks))
    categories = [
        'hammer_throw',
        'long_jump',
        'shot_put',
        'discus_throw',
        'basketball_layup',
        'bowling',
        'clean_and_jerk',
        'diving_platform_10m',
        'diving_springboard_3m',
        'snatch',
        'high_jump',
        'javelin_throw',
        'pole_vault',
        'hammer_throw',
        'tennis_serve',
        'triple_jump',
        'vault'
    ]

    categories = np.asarray(categories)
    chunk_size = len(categories) / num_chunks
    chunk_start = range(0, len(categories), chunk_size)

    cur_cats = list()
    for chunk_id in chunk_ids:
        assert 0 <= chunk_id < num_chunks
        cur_cats.extend(categories[chunk_start[chunk_id]:chunk_start[chunk_id] + chunk_size])
    print cur_cats
    for cat in cur_cats:
        train.main([gpu_id_str, cat])

    # p = multiprocessing.Pool(n_workers)
    # p.map(train_cliquecnn_reassign.main, [(gpu_id_str, cat) for cat in categories], 1)
    # p.close()
    # p.join()

    # Parallel(n_jobs=n_jobs)(delayed(train_cliquecnn_reassign.main)([gpu_id_str, cat]) for cat in categories)
    # Parallel(n_jobs=n_jobs)(delayed(eval.olympicsports.compute_sim.main)([gpu_id_str, cat])
    #                    for cat in tqdm(categories))
    # Parallel(n_jobs=n_jobs)(delayed(train_and_compute_sim)([gpu_id_str, cat]) for cat in categories)

