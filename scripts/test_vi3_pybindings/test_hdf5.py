import sys
import os
import numpy as np
from sklearn.datasets import load_digits

inst_path = '/home/consti/Work/projects_phd/ilastik-hackathon/inst/lib/python2.7/dist-packages'
assert os.path.exists(inst_path)
sys.path.append(inst_path)
import vigra

rf3 = vigra.learning.RandomForest3

digits = load_digits()
data = digits.data
labels = digits.target
shuffle = np.random.permutation(data.shape[0])

X_train = data[shuffle[:400]].astype('float32')
Y_train = labels[shuffle[:400]].astype('uint32')

X_test = data[shuffle[400:]].astype('float32')
Y_test = labels[shuffle[400:]].astype('uint32')

def accuracy(prediction):
    assert prediction.shape == Y_test.shape
    return np.sum(prediction == Y_test) / float(Y_test.shape[0])

def test_hdf5_serialization():
    rf = rf3(X_train, Y_train)
    rf.writeHDF5('./tmp.h5', 'data')
    rf_loaded = rf3('./tmp.h5', 'data')
    probs = rf_loaded.predictProbabilities(X_test)
    prediction = np.argmax(probs, axis = 1)
    print "Accuracy predict probabilities"
    print accuracy(prediction)
    os.remove('./tmp.h5')


if __name__ == '__main__':
    test_hdf5_serialization()
