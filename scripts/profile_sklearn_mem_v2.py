import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf_sk
from memory_profiler import memory_usage

print sklearn.__version__ # check the version directly

# I haven't figured out how to pass arguments to the function that
# is profiled with memory_usage. Hence, all relevant variables are
# declared globally and then also set globally in the functions
X_train = []
Y_train = []
X = []
rf = None

def load_ilastik_data(path_to_il_train_features,
        path_to_il_train_labels,
        path_to_il_test_features):

    # global workaround
    global X_train
    global Y_train
    global X

    import h5py
    with h5py.File(path_to_il_train_features) as f:
        X_train = f['data'][:].astype('float64')
    with h5py.File(path_to_il_train_labels) as f:
        Y_train = f['data'][:].astype('int64')
    with h5py.File(path_to_il_test_features) as f:
        X = f['data'][:].astype('float64')
    shape = X.shape
    X = X.reshape((shape[0]*shape[1]*shape[2],shape[3]))
    print "Loading ilastik data"
    print "Types:", type(X), type(X_train), type(Y_train)
    print "Dtypes:", X_train.dtype, Y_train.dtype, X.dtype
    print "Shapes:", X_train.shape, Y_train.shape, X.shape
    print "Unique labels:", np.unique(Y_train)


# TODO load some sklearn dataset
def load_digits_data():
    from sklearn.datasets import load_digits

    # global workaround
    global X_train
    global Y_train
    global X

    digits = load_digits()
    data = digits.data
    labels = digits.target

    # shuffle the data to have multiple classes in the training data
    shuffle = np.random.permutation(data.shape[0])
    X_train = data[shuffle[:200]]
    Y_train = labels[shuffle[:200]]
    X = data[shuffle[200:]]
    print "Loading digits data"
    print "Types:", type(X), type(X_train), type(Y_train)
    print "Dtypes:", X_train.dtype, Y_train.dtype, X.dtype
    print "Shapes:", X_train.shape, Y_train.shape, X.shape
    print "Unique labels:", np.unique(Y_train)


def train_sklearn(n_trees, n_threads):
    global rf # global workaround...
    rf = rf_sk(n_estimators = n_trees, n_jobs = n_threads)
    rf.fit(X_train, Y_train)


def predict_sklearn():
    rf.predict_proba(X)


def monitor_ram(n_trees, n_threads):
    train_sklearn(n_trees, n_threads)
    mem_usage = memory_usage(predict_sklearn)
    max_ram = max(mem_usage)
    return max_ram


def test_ram_consumption():
    mem_dict = {}
    tree_tests = (1,2,4,5,10,25)
    #thread_tests = (1,2,4,8,10,20)
    n_threads = 1
    for n_trees in tree_tests:
        max_ram = monitor_ram(n_trees, n_threads)
        print "n_estimators: %i: max-Ram usage: %.2f MB" % (n_trees, max_ram)


if __name__ == '__main__':
    load_ilastik_data('../training_data/features_train.h5',
            '../training_data/labels_train.h5',
            '../training_data/features_test.h5')
    #load_digits_data()

    test_ram_consumption()
