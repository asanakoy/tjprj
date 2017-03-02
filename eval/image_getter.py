# Copyright (c) 2016 Artsiom Sanakoyeu
import h5py
import numpy as np
import PIL
from PIL import Image


class ImageGetterFromMat:
    """
    Class for loading batches by index from mat file
    """
    def __init__(self, mat_path, load_all_in_memory=False, rgb_batch=False):
        dataset = h5py.File(mat_path, 'r')
        has_images_mat = 'images_mat' in dataset
        has_images = 'images' in dataset
        if has_images_mat and has_images:
            raise Exception('have two image matrix!')
        if has_images_mat:
            self.images_ref = dataset['images_mat']
        else:
            self.images_ref = dataset['images']
        if load_all_in_memory:
            self.images_ref = self.images_ref[...]
        self.rgb_batch = rgb_batch

    def total_num_images(self):
        return self.images_ref.shape[0]

    def get_batch(self, indxs, resize_shape=None, mean=None):
        """
        Get batch by the indices of the images.
        :param indxs: numeric indices of the images to include in the batch
        :param resize_shape:
        :param mean: must be HxWxC RGB
        :return: batch of images as np.array NxHxWxC with:
            RGB channel order if self.rgb_batch == True
            BGR channel order otherwise
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
        if not self.rgb_batch:
            batch = batch[:, :, :, (2, 1, 0)]  # reorder channels RGB -> BGR
        return batch


class ImageGetterFromPaths:
    """
    Class for loading batches by index from disk by paths
    """

    def __init__(self, image_paths, im_shape, rgb_batch=False):
        """
        :param image_paths: list of full pathes
        :param im_shape: default im_shape to use when reading images
        """
        assert len(im_shape) == 2, 'im_shape must be of len 2: (h, w)!'
        self.image_paths = image_paths
        self.im_shape = im_shape
        self.rgb_batch = rgb_batch

    def total_num_images(self):
        return len(self.image_paths)

    def get_batch(self, indxs, resize_shape=None, mean=None):
        """
        Get batch by the indices of the images.
        :param indxs: numeric indices of the images to include in the batch
        :param resize_shape: resize images to this shape.
                             If None, resize to self.im_shape.
        :param mean: must be HxWxC RGB
        :return: batch of images as np.array NxHxWxC with:
            RGB channel order if self.rgb_batch == True
            BGR channel order otherwise
        """
        assert len(resize_shape) == 2, 'resize_shape must be of len 2: (h, w)!'
        assert mean is None or (len(mean.shape) == 3 and mean.shape[2] == 3)

        if resize_shape is None:
            resize_shape = self.im_shape

        # NxHxWxC RGB matrix
        batch = np.zeros((len(indxs),) + resize_shape + (3,), dtype=np.float32)
        for i, image_idx in enumerate(indxs):
            image = Image.open(self.image_paths[image_idx]).resize(resize_shape,
                                                                   PIL.Image.ANTIALIAS)
            batch[i, ...] = np.asarray(image, dtype=np.float32)

        if mean is not None:
            batch -= np.tile(mean, (batch.shape[0], 1, 1, 1))
        if not self.rgb_batch:
            batch = batch[:, :, :, (2, 1, 0)]  # reorder channels RGB -> BGR
        return batch
