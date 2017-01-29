import vigra
import h5py
import numpy as np

from volumina_viewer import volumina_n_layer

raw_file = '/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit_squeezed.h5'
prediction_file = '../results/prediction.h5'
test_slice = np.s_[2500:2700,2600:2800,0:200]

prediction = vigra.readHDF5(prediction_file, 'data')

with h5py.File(raw_file) as raw_f:
    raw_ds = raw_f['data']
    raw = raw_ds[test_slice].astype('float32')

assert raw.shape == prediction.shape[:-1], str(raw.shape) + " , " + str(prediction.shape)

volumina_n_layer([raw, prediction])
