import sys
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier as rf_sk
import numpy as np
import time
from concurrent import futures
sys.path.append('/home/consti/Work/projects_phd/ilastik-hackathon/inst/lib/python2.7/dist-packages')
import vigra

X_train = vigra.readHDF5('./training_data/annas_features.h5', 'data')
Y_train = vigra.readHDF5('./training_data/annas_labels.h5', 'data')

X = vigra.readHDF5('./training_data/features_test', 'data')
shape = X.shape
X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))

rf2 = vigra.learning.RandomForest

# number of repetitions
N = 1
# we keep n_trees fixed
n_trees = 100

# do a grid search over min_split_node and max_depth
min_split_node_vals = [1,2,5,10]
max_depth_vals     = [-1,4,6,8,12]

def learn_rf(min_nodes, max_depth, n_threads=2):

    def learn_sub_rf(Xtr, Ytr, max_depth, min_nodes, n_sub):
        rf = rf2(treeCount = n_sub, min_split_node_size = min_nodes)
        rf.learnRF(Xtr, Ytr, maxDepth = max_depth)
        return rf

    sub_trees  = n_threads * [n_trees / n_threads]
    sub_trees[0] += n_trees % n_threads

    with futures.ThreadPoolExecutor(n_worker = n_threads) as executor:
        tasks = []
        for t in xrange(n_threads):
            tasks.append( executor.submit(
                learn_sub_rf,
                X_train,
                Y_train,
                max_depth,
                min_nodes,
                n_subtrees[t]) )
        rfs = [tt.result() for tt in tasks]
    return rfs

def predict_rf(rfs, n_threads=2):

    with futures.ThreadPoolExecutor(n_worker = n_threads) as executor:
        tasks = []
        for t in xrange(n_threads):
            tasks.append( executor.submit( rfs[t].predictProbabilities, X) )
        sub_probs = [rfs[ii].treeCount() * tt.result() for ii, tt in enumerate(tasks)]
        probs = np.sum(sub_probs, axis = 0)
        probs /= n_trees

    return probs

# TODO need evaluation
def grid_search():
    for min_node in min_split_node_vals:
        for max_depth in max_depth_vals:
            t_train = time.time()
            rfs = learn_rf(min_node, max_depth, 4)
            t_train = time.time() - t_train

            t_test = time.time()
            predict_rfs(rfs, 4)
            t_test = time.time() - t_test
