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
Y_train = vigra.readHDF5('./training_data/annas_labels.h5', 'data')

X = vigra.readHDF5('./training_data/features_test.h5', 'data')
shape = X.shape
X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))

rf2 = vigra.learning.RandomForest
rf3 = vigra.learning.RandomForest3

# number of repetitions
N = 1
# parameter, for the rest the defaults should agree
n_trees = 100
min_split_node = 2

def predict_vi2(n_threads=1):
    times = []
    if n_threads == 1:
        rf = rf2(treeCount = n_trees, min_split_node_size = min_split_node)
        rf.learnRF(X_train, Y_train[:,None].astype('uint32'))
        for _ in xrange(N):
            t0 = time.time()
            probs = rf.predictProbabilities(X)
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

        with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
            tasks = []
            for t in xrange(n_threads):
                tasks.append(executor.submit(
                    train_sub_rf,
                    X_train,
                    Y_train[:,None].astype('uint32'),
                    sub_trees[t]))
            rfs = [tt.result() for tt in tasks]

        for _ in xrange(N):
            t0 = time.time()
            with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                tasks = []
                for t in xrange(n_threads):
                    tasks.append( executor.submit( rfs[t].predictProbabilities, X) )
                sub_probs = [sub_trees[ii] * tt.result() for ii, tt in enumerate(tasks)]
                probs = np.sum(sub_probs, axis = 0) # ?!
                probs /= n_trees
            times.append(time.time() - t0)

    return np.mean(times), np.std(times)

def predict_vi3(n_threads=1):
    times = []
    rf = rf3(X_train, Y_train.astype('uint32'),
            treeCount = n_trees, min_split_node_size = min_split_node,
            n_threads = n_threads)
    for _ in xrange(N):
        t0 = time.time()
        probs = rf.predictProbabilities(X, n_threads = n_threads)
        times.append(time.time() - t0)
    return np.mean(times), np.std(times)

def predict_sk(n_threads=1):
    rf = rf_sk(n_estimators = n_trees, min_samples_split = min_split_node,
            n_jobs = n_threads)
    rf.fit(X_train,Y_train)
    times = []
    for _ in xrange(N):
        t0 = time.time()
        probs = rf.predict_proba(X)
        times.append(time.time() - t0)
    return np.mean(times), np.std(times)

def compare_predict_rfs():
    res_dict = {}
    threads = (1,2,4,6,8)
    print "Start prediction benchmarks"
    # vi2
    res_dict["vi2"] = {}
    for n_threads in threads:
        t_vi2, std_vi2 = predict_vi2(n_threads)
        res_dict["vi2"][n_threads] = (t_vi2, std_vi2)
        print "Predicting vigra rf2 with %i threads in %f +- %f s" % (n_threads, t_vi2, std_vi2)
    # vi3
    res_dict["vi3"] = {}
    for n_threads in threads:
        t_vi3, std_vi3 = predict_vi3(n_threads)
        res_dict["vi3"][n_threads] = (t_vi3, std_vi3)
        print "Predicting vigra rf3 with %i threads in %f +- %f s" % (n_threads, t_vi3, std_vi3)
    # TODO FIXME This is insanely RAM-hungry, write sklearn issue!
    # sk
    #res_dict["sk"] = {}
    #for n_threads in threads:
    #    t_sk, std_sk = predict_sk(n_threads)
    #    res_dict["sk"][n_threads] = (t_sk, std_sk)
    #    print "Predicting sklearn rf with %i threads in %f +- %f s" % (n_threads, t_sk, std_sk)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    with open('./results/benchmarks_prediction.pkl', 'w') as f:
        pickle.dump(res_dict, f)

if __name__ == '__main__':
    compare_predict_rfs()
