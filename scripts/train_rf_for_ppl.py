import sys
import os
import numpy as np
from concurrent import futures
import cPickle as pickle

sys.path.append('/home/constantin/Work/my_projects/ilastik-hackathon/inst/lib/python2.7/dist-packages')
import vigra

X = vigra.readHDF5('../training_data/annas_features.h5', 'data')
Y = vigra.readHDF5('../training_data/annas_labels.h5', 'data')[:,None].astype('uint32')

rf2 = vigra.learning.RandomForest

# parameter, for the rest the defaults should agree
n_trees = 100
min_split_node = 2

# parallelisation ilastik-style
def train_rf(max_depth, n_threads):

    if n_threads == 1:
        rf = rf2(treeCount = n_trees, min_split_node_size = min_split_node)
        rf.learnRF(X, Y, maxDepth = max_depth)
        rfs = [rf]

    else:
        sub_trees  = n_threads * [n_trees / n_threads]
        remaining_trees = n_trees % n_threads
        for extra_tree in xrange(remaining_trees):
            sub_trees[extra_tree] += 1

        def train_sub_rf(Xtr, Ytr, n_subs):
            rf = rf2(treeCount = n_subs, min_split_node_size = min_split_node)
            rf.learnRF(Xtr, Ytr, maxDepth = max_depth)
            return rf

        with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
            tasks = []
            for t in xrange(n_threads):
                tasks.append(executor.submit(
                    train_sub_rf, X, Y, sub_trees[t]))
            rfs = [tt.result() for tt in tasks]

    return rfs


def train_and_save_rf(save_folder, max_depth, n_threads):
    save_name = "RandomForest_MaxDepth%i_NThreads%i" % (max_depth, n_threads)
    save_path = os.path.join(save_folder, save_name)

    rfs = train_rf(max_depth, n_threads)
    for i, rf in enumerate(rfs):
        key = "rf_" + str(i)
        rf.writeHDF5(save_path, key)


if __name__ == '__main__':
    save_folder = '/home/consti/Work/data_neuro/ilastik_hackathon/'
    train_and_save_rf(save_folder, 8, 1)
    train_and_save_rf(save_folder, 8, 8)
    train_and_save_rf(save_folder, -1, 8)


