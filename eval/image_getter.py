# Copyright (c) 2016 Artsiom Sanakoyeu
import h5py
import numpy as np
import PIL
from PIL import Image


class ImageGetterFromMat:
    """
    Class for loading batches by index from mat file
    """
    def __init__(self, mat_path):
        dataset = h5py.File(mat_path, 'r')
        self.images_ref = dataset['images_mat']

    def total_num_images(self):
        return self.images_ref.shape[0]

    def get_batch(self, indxs, resize_shape=None, mean=None):
        """
        Get batch by the indices of the images.
        :param indxs: numeric indices of the images to include in the batch
        :param resize_shape:
        :param mean: must be HxWxC RGB
        :return: batch of images as np.array NxHxWxC with BGR channel order
        """
        assert len(resize_shape) == 2, 'resize_shape must be of len 2: (h, w)!'
        assert mean is None or (len(mean.shape) == 3 and mean.shape[2] == 3)
        batch = self.images_ref[indxs, :, :, :][...]  # matlab format CxWxH x N
        batch = batch.transpose((0, 3, 2, 1))  # N x HxWxC matrix

        if resize_shape is not None:
            resized_batch = np.zeros((batch.shape[0],) + resize_shape + (3,), dtype=np.float32)
            for i in xrange(batch.shape[0]):
                image = np.asarray(
                    Image.fromarray(batch[i, ...]).resize(resize_shape, PIL.Image.ANTIALIAS))
                resized_batch[i, ...] = image
            batch = resized_batch

        batch = np.asarray(batch, dtype=np.float32)
        if mean is not None:
            batch -= np.tile(mean, (batch.shape[0], 1, 1, 1))
        batch = batch[:, :, :, (2, 1, 0)]  # reorder channels RGB -> BGR
        return batch
