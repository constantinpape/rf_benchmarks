from sklearn.ensemble import RandomForestClassifier as rf_sk
import sklearn.datasets
import numpy as np
from memory_profiler import memory_usage
import cPickle as pickle

# Needs to be global because I haven't figured out how to pass arguments to mem_profiler
X_train = []
Y_train = []
X = []

def load_ilastik_data(path_to_il_train_features,
        path_to_il_train_labels,
        path_to_il_test_features):

    global X_train
    global Y_train
    global X

    import h5py
    with h5py.File(path_to_il_train_features) as f:
        X_train = f['data'][:]
    with h5py.File(path_to_il_train_labels) as f:
        Y_train = f['data'][:]
    with h5py.File(path_to_il_test_features) as f:
        X = f['data'][:]



X_train = vigra.readHDF5('./training_data/annas_features.h5', 'data')
Y_train = vigra.readHDF5('./training_data/annas_labels.h5', 'data')


X = vigra.readHDF5('./training_data/features_test.h5', 'data')
shape = X.shape
X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))

n_trees   = -1
n_threads = -1

def run_sklearn():
    assert n_trees != -1, n_trees
    assert n_threads != -1, n_threads
    print "N-trees:", n_trees, "N-threads:", n_threads
    rf = rf_sk(n_estimators = n_trees, n_jobs = n_threads)
    rf.fit(X_train, Y_train)
    print rf.classes_
    rf.predict_proba(X)

def monitor_ram(n_tr, n_th):
    global n_trees
    n_trees = n_tr
    global n_threads
    n_threads = n_th
    mem_usage = memory_usage(run_sklearn)
    max_ram = max(mem_usage)
    return max_ram

def test_ram_consumption():
    mem_dict = {}
    tree_tests = (5,10,25,50,100,200)
    thread_tests = (1,2,4,8,10,20)
    for n_tr in tree_tests:
        for n_th in thread_tests:
            max_ram = monitor_ram(n_tr, n_th)
            print "Max-Ram:", max_ram
            quit()
            mem_dict[(n_tr, n_th)] = max_ram
    with open('./results/sklearn_ram.pkl', 'w') as f:
        pickle.dump(mem_dict,f)

if __name__ == '__main__':
    test_ram_consumption()
