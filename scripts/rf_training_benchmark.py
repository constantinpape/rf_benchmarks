import sys
import os
from sklearn.ensemble import RandomForestClassifier as rf_sk
import numpy as np
import time
from concurrent import futures
import cPickle as pickle
sys.path.append('/home/constantin/Work/my_projects/ilastik-hackathon/inst/lib/python2.7/dist-packages')
import vigra

X = vigra.readHDF5('../training_data/annas_features.h5', 'data')
Y = vigra.readHDF5('../training_data/annas_labels.h5', 'data')

rf2 = vigra.learning.RandomForest
rf3 = vigra.learning.RandomForest3

# number of repetitions
N = 20
# parameter, for the rest the defaults should agree
n_trees = 100
min_split_node = 2

# we always construct the rf in the loop because rf3 does not have an empty constructor

# parallelisation ilastik-style
def train_vi2(n_threads=1):
    times = []
    if n_threads == 1:
        for _ in xrange(N):
            t0 = time.time()
            rf = rf2(treeCount = n_trees, min_split_node_size = min_split_node)
            rf.learnRF(X, Y[:,None].astype('uint32'))
            times.append(time.time() - t0)
    else:
        sub_trees  = n_threads * [n_trees / n_threads]
        remaining_trees = n_trees % n_threads
        for extra_tree in xrange(remaining_trees):
            sub_trees[extra_tree] += 1
        def train_sub_rf(Xtr, Ytr, n_subs):
            rf = rf2(treeCount = n_subs, min_split_node_size = min_split_node)
            rf.learnRF(Xtr, Ytr)
            return rf
        for _ in xrange(N):
            t0 = time.time()
            with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                tasks = []
                for t in xrange(n_threads):
                    tasks.append(executor.submit(
                        train_sub_rf,
                        X,
                        Y[:,None].astype('uint32'),
                        sub_trees[t]))
                rfs = [tt.result() for tt in tasks]
            times.append(time.time() - t0)
    return np.mean(times), np.std(times)

def train_vi3(n_threads=1):
    times = []
    for _ in xrange(N):
        t0 = time.time()
        rf = rf3(X, Y.astype('uint32'),
                treeCount = n_trees, min_split_node_size = min_split_node,
                n_threads = n_threads)
        times.append(time.time() - t0)
    return np.mean(times), np.std(times)

def train_sk(n_threads=1):
    times = []
    for _ in xrange(N):
        t0 = time.time()
        rf = rf_sk(n_estimators = n_trees, min_samples_split = min_split_node,
                n_jobs = n_threads)
        rf.fit(X,Y)
        times.append(time.time() - t0)
    return np.mean(times), np.std(times)

def compare_train_rfs():
    res_dict = {}
    #threads = (1,2,4,6,8)
    threads = (1,2,4,6,8,10,20,30,40)
    print "Start training benchmarks"
    # vi2
    res_dict["vi2"] = {}
    for n_threads in threads:
        t_vi2, std_vi2 = train_vi2(n_threads)
        res_dict["vi2"][n_threads] = (t_vi2, std_vi2)
        print "Training vigra rf2 with %i threads in %f +- %f s" % (n_threads, t_vi2, std_vi2)
    # vi3
    res_dict["vi3"] = {}
    for n_threads in threads:
        t_vi3, std_vi3 = train_vi3(n_threads)
        res_dict["vi3"][n_threads] = (t_vi3, std_vi3)
        print "Training vigra rf3 with %i threads in %f +- %f s" % (n_threads, t_vi3, std_vi3)
    # sk
    res_dict["sk"] = {}
    for n_threads in threads:
        t_sk, std_sk = train_sk(n_threads)
        res_dict["sk"][n_threads] = (t_sk, std_sk)
        print "Training sklearn rf with %i threads in %f +- %f s" % (n_threads, t_sk, std_sk)

    if not os.path.exists('../results'):
        os.mkdir('../results')
    with open('../results/benchmarks_training.pkl', 'w') as f:
        pickle.dump(res_dict, f)

if __name__ == '__main__':
    compare_train_rfs()
