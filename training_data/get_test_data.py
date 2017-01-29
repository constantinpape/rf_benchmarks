import vigra
import h5py
import numpy as np

raw_file = '/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit_squeezed.h5'

filters = ["gaussianSmoothing", "laplacianOfGaussian", "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues"]

sigmas = { "gaussianSmoothing" : (0.3,1.0,1.6,3.5,5.0,10.0),
        "laplacianOfGaussian"  : (1.0,3.5,10.0),
        "gaussianGradientMagnitude" : (1.6,5.),
        "hessianOfGaussianEigenvalues" : (1.5,5.) }

halo = 35

test_slice = np.s_[2500:2700,2600:2800,0:200]

with h5py.File(raw_file) as raw_f:

    raw_ds = raw_f['data']

    raw = raw_ds[test_slice].astype('float32')
    shape = raw.shape

    features = np.zeros(shape + (17,), dtype = 'float32')

    channel = 0
    for filterName in filters:
        print "Applying %s" % filterName
        filter_ = eval("vigra.filters." + filterName)
        for sigma in sigmas[filterName]:
            print "with sigma %f" % sigma
            response = filter_(raw, sigma)
            if(response.ndim == 3):
                response = response[...,None]
            for c in xrange(response.shape[-1]):
                features[...,channel] = response[...,c]
                channel += 1

vigra.writeHDF5(features ,'./features_test', 'data')
