import vigra
import h5py
import numpy as np
import re

il_file = './hackathon_flyem.ilp'
raw_file = '/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit_squeezed.h5'
path_to_labels = 'PixelClassification/LabelSets/labels000/'

n_blocks = 235
halo     = 35

features = []
labels =   []

filters = ["gaussianSmoothing", "laplacianOfGaussian", "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues"]

sigmas = { "gaussianSmoothing" : (0.3,1.0,1.6,3.5,5.0,10.0),
        "laplacianOfGaussian"  : (1.0,3.5,10.0),
        "gaussianGradientMagnitude" : (1.6,5.),
        "hessianOfGaussianEigenvalues" : (1.5,5.) }


with h5py.File(il_file) as il_proj, \
    h5py.File(raw_file) as raw_f:

    raw = raw_f['data']
    shape = raw.shape

    for i in xrange(n_blocks):

        print i

        block_path = path_to_labels + ("block%04i" % i)
        labelset = il_proj[block_path]
        indices = labelset.attrs["blockSlice"]
        indices = list( map( lambda x : int(x), re.findall(r'\b\d+\b', indices) ) )

        slicing = np.s_[max(indices[0]-halo,0):min(indices[1]+halo,shape[0]),
                        max(indices[2]-halo,0):min(indices[3]+halo,shape[1]),
                        max(indices[4]-halo,0):min(indices[5]+halo,shape[2])]

        x_diff = indices[1] - indices[0]
        x_left_offset  = indices[0] - max(indices[0] - halo,0)
        x_right_offset = min(indices[1] + halo,shape[0]) - indices[1] + x_diff

        y_diff = indices[3] - indices[2]
        y_left_offset  = indices[2] - max(indices[2] - halo,0)
        y_right_offset = min(indices[3] + halo,shape[1]) - indices[3] + y_diff

        z_diff = indices[5] - indices[4]
        z_left_offset  = indices[4] - max(indices[4] - halo,0)
        z_right_offset = min(indices[5] + halo,shape[2]) - indices[5] + z_diff

        label_slicing = np.s_[x_left_offset:x_right_offset,
                y_left_offset:y_right_offset,
                z_left_offset:z_right_offset]

        # get labels for this block
        label_array = labelset[:][...,0]
        labeled = label_array != 0
        labels_block = label_array[labeled]
        labels.extend(list(labels_block))

        raw_array = raw[slicing].astype('float32')

        # get features for this block
        features_block = np.zeros( (labels_block.shape[0], 17), dtype = 'float32' )
        channel = 0
        for filtername in filters:
            filter_ = eval("vigra.filters." + filtername)
            for sigma in sigmas[filtername]:
                response = filter_(raw_array, sigma)
                response = response[label_slicing]
                if response.ndim == 3:
                    response = response[...,None]
                assert response.shape[:-1] == label_array.shape, str(response.shape) + str(label_array.shape)
                for c in range(response.shape[3]):
                    features_block[:,channel] = response[...,c][labeled]
                    channel += 1
        features.append(features_block)

labels = np.array(labels, dtype = 'uint8')
features = np.concatenate(features, axis = 0)

print labels.shape
print features.shape
print np.unique(labels)
labels -= labels.min()
print np.unique(labels)

vigra.writeHDF5(features, "annas_features.h5", "data")
vigra.writeHDF5(labels, "annas_labels.h5", "data")
