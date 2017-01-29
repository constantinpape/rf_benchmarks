import vigra
import h5py
import numpy as np
import os

from volumina_viewer import volumina_n_layer

raw_file = '/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit_squeezed.h5'
test_slice = np.s_[2500:2700,2600:2800,0:200]

def check_single_prediction():
    prediction_file = '../results/prediction.h5'

    prediction = vigra.readHDF5(prediction_file, 'data')

    with h5py.File(raw_file) as raw_f:
        raw_ds = raw_f['data']
        raw = raw_ds[test_slice].astype('float32')

    assert raw.shape == prediction.shape[:-1], str(raw.shape) + " , " + str(prediction.shape)

    volumina_n_layer([raw, prediction])

def check_mulit_predictions(*args):

    predictions = []
    for pred_path in args:
        assert os.path.exists(pred_path)
        predictions.append( vigra.readHDF5(pred_path, 'data') )

    with h5py.File(raw_file) as raw_f:
        raw_ds = raw_f['data']
        raw = raw_ds[test_slice].astype('float32')

    for pred in predictions:
        assert raw.shape == pred.shape[:-1], str(raw.shape) + " , " + str(pred.shape)

    volumina_n_layer([raw] + predictions)

if __name__ == '__main__':
    check_mulit_predictions('../results/prediction.h5',
            '../results/prediction_1_-1.h5',
            '../results/prediction_1_4.h5',
            '../results/prediction_1_6.h5')
