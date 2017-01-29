import sys
import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier as rf_sk
import numpy as np
import time
from concurrent import futures
sys.path.append('/home/consti/Work/my_projects/ilastik-hackathon/inst/lib/python2.7/dist-packages')
import vigra

X_train = vigra.readHDF5('./training_data/annas_features.h5', 'data')
Y_train = vigra.readHDF5('./training_data/annas_labels.h5', 'data')[:,None].astype('uint32')

X = vigra.readHDF5('./training_data/features_test.h5', 'data')
shape = X.shape
X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))

rf2 = vigra.learning.RandomForest

# we keep n_trees fixed
n_trees = 100
min_size_node = 2

# do a grid search over min_split_node and max_depth
min_size_leaf_vals = [0,2,5,10,15,20]
max_depth_vals     = [8,10,12,15,-1]


def learn_rf(max_depth, min_size_leaf, n_threads=2):

    def learn_sub_rf(Xtr, Ytr, max_depth, min_size_leaf, n_sub):
        rf = rf2(treeCount = n_sub, min_split_node_size = min_size_node)
        rf.learnRF(Xtr, Ytr, maxDepth = max_depth, minSize = min_size_leaf)
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
                min_size_leaf,
                subtrees[t]) )
        rfs = [tt.result() for tt in tasks]
    return rfs


def predict_rf(rfs, n_threads=2, save = False, save_name = "prediction.h5"):

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
                    './results/' + save_name, 'data')
    return probs


def eval_pmap(probs, reference_pmap):
    assert probs.shape == reference_pmap.shape
    assert probs.ndim == 2
    max_validation = np.argmax(probs, axis = 1)
    max_reference  = np.argmax(reference_pmap, axis = 1)
    return np.sum(max_validation == max_reference) / float(max_reference.shape[0])


def grid_search(N, n_threads, save=False):

    if save:
        assert N == 1

    reference_pmap = vigra.readHDF5('./results/prediction.h5', 'data')
    reference_pmap = reference_pmap.reshape((shape[0]*shape[1]*shape[2],4))

    res_dict = {}
    for max_depth in max_depth_vals:
        for min_leaf_size in min_size_leaf_vals:

            print "Eval for: ", max_depth, min_leaf_size

            t_train = []
            t_test  = []
            acc     = []

            for _ in xrange(N):
                t0 = time.time()
                rfs = learn_rf(max_depth, min_leaf_size, n_threads)
                t_train.append(time.time() - t0)
                t1 = time.time()
                probs  = predict_rf(rfs, n_threads, save,
                        "prediction_depth%i_split%i.h5" % (max_depth,min_leaf_size))
                t_test.append(time.time() - t1)
                acc.append(eval_pmap(probs, reference_pmap))

            t_train_m, t_train_std = np.mean(t_train), np.std(t_train)
            t_test_m, t_test_std = np.mean(t_test), np.std(t_test)
            acc_m, acc_std = np.mean(acc), np.std(acc)

            print "Train Time:", t_train_m, "+-", t_train_std ,"s"
            print "Prediction Time", t_test_m, "+-", t_test_std ,"s"
            print "Accuracy", acc_m, "+-", acc_std, "s"
            res_dict[(max_depth, min_leaf_size)] = (t_train_m, t_train_std, t_test_m, t_test_std, acc_m,acc_std)

    if not save:
        if not os.path.exists('./results'):
            os.mkdir('./results')
        with open('./results/benchmarks_gridsearch.pkl', 'w') as f:
            pickle.dump(res_dict, f)


if __name__ == '__main__':
    #rfs = learn_rf(-1, 0, 4)
    #pmap = predict_rf(rfs, 4, True)

    grid_search(15, 4)
