import sys
import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier as rf_sk
import numpy as np
import time
from concurrent import futures
sys.path.append('/home/consti/Work/projects_phd/ilastik-hackathon/inst/lib/python2.7/dist-packages')
import vigra

X_train = vigra.readHDF5('./training_data/annas_features.h5', 'data')
Y_train = vigra.readHDF5('./training_data/annas_labels.h5', 'data')[:,None].astype('uint32')

X = vigra.readHDF5('./training_data/features_test.h5', 'data')
shape = X.shape
X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))

rf2 = vigra.learning.RandomForest

# number of repetitions
N = 1
# we keep n_trees fixed
n_trees = 100

# do a grid search over min_split_node and max_depth
min_split_node_vals = [1,2,5,10]
max_depth_vals      = [-1,4,6,8,12]


def learn_rf(min_nodes, max_depth, n_threads=2):

    def learn_sub_rf(Xtr, Ytr, max_depth, min_nodes, n_sub):
        rf = rf2(treeCount = n_sub, min_split_node_size = min_nodes)
        rf.learnRF(Xtr, Ytr, maxDepth = max_depth)
        return rf

    subtrees  = n_threads * [n_trees / n_threads]
    remaining_trees = n_trees % n_threads
    for extra_tree in xrange(remaining_trees):
        sub_trees[extra_tree] += 1

    with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
        tasks = []
        for t in xrange(n_threads):
            tasks.append( executor.submit(
                learn_sub_rf,
                X_train,
                Y_train,
                max_depth,
                min_nodes,
                subtrees[t]) )
        rfs = [tt.result() for tt in tasks]
    return rfs


def predict_rf(rfs, n_threads=2, save = False):

    with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
        tasks = []
        for t in xrange(n_threads):
            tasks.append( executor.submit( rfs[t].predictProbabilities, X) )
        sub_probs = [rfs[ii].treeCount() * tt.result() for ii, tt in enumerate(tasks)]
        probs = np.sum(sub_probs, axis = 0)
        probs /= n_trees

        if save:
            if not os.path.exists('./results'):
                os.mkdir('./results')
            vigra.writeHDF5(probs.reshape( (shape[0], shape[1], shape[2], 4) ),
                    './results/prediction.h5', 'data')
    return probs


def eval_pmap(probs, reference_pmap):
    assert probs.shape == reference_pmap.shape
    max_validation = np.argmax(probs, axis = 0)
    max_reference  = np.argmax(reference_pmap, axis = 0)
    agree = max_validation == max_reference
    return np.sum(agree) / float(probs.shape[0])


def grid_search():

    reference_pmap = vigra.readHDF5('./results/prediction.h5', 'data')
    reference_pmap = reference_pmap.reshape((shape[0]*shape[1]*shape[2],4))

    res_dict = {}
    for min_node in min_split_node_vals:
        for max_depth in max_depth_vals:
            print "Eval for: ", min_node, max_depth
            t_train = time.time()
            rfs = learn_rf(min_node, max_depth, 4)
            t_train = time.time() - t_train
            t_test = time.time()
            probs  = predict_rf(rfs, 4)
            t_test = time.time() - t_test
            acc = eval_pmap(probs, reference_pmap)
            res_dict[(min_node, max_depth)] = (t_train, t_test, acc)

            print "Train Time:", t_train, "s"
            print "Prediction Time", t_test, "s"
            print "Accuracy", acc

    if not os.path.exists('./results'):
        os.mkdir('./results')
    with open('./results/benchmarks_gridsearch.pkl', 'w') as f:
        pickle.dump(res_dict, f)


if __name__ == '__main__':
    # for eval and validation
    #rfs = learn_rf(1, -1, 4)
    #pmap = predict_rf(rfs, 4, True)
    grid_search()
