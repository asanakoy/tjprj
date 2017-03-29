# Artsiom Sanakoyeu, 2016

from chainer.dataset import dataset_mixin
import cv2 as cv
import logging
import numpy as np
import os
from tqdm import tqdm
import glob


class RandomClipSubsampleType:
    CONSECUTIVE = 0
    WITH_PERIOD = 1
    RANDOM_CHOOSE = 2


class SequenceDataset(dataset_mixin.DatasetMixin):
    def __init__(self, sequences_by_category,
                 images_dir,
                 image_size,
                 crop_size,
                 random_clip_subsample_type=None,
                 fliplr=False,
                 num_frames_in_clip=16,
                 should_downscale_images=False,
                 downscale_height=227,
                 img_ext='.png',
                 seed=None):
        for key, val in locals().items():
            setattr(self, key, val)
        if len(image_size) != 2:
            raise ValueError('Image size must be 2D')
        if len(crop_size) != 2:
            raise ValueError('Image size must be 2D')
        print 'SequenceDataset:: Random seed={}'.format(seed)
        self.rs = np.random.RandomState(seed)
        self.downscale_factor = None
        self.sequences = dict()
        self.index2seqid = None
        self.load_sequences()
        logging.info('{} is ready'.format(sequences_by_category))

    def load_sequences(self):
        """
        Load all sequences in memory
        """
        self.sequences = dict()
        print('Reading images from {}'.format(self.images_dir))
        if self.should_downscale_images:
            print('Downscale images to the height {}px'.format(self.downscale_height))
        for category_name, sequences in self.sequences_by_category.iteritems():
            for seq_name in tqdm(sequences, desc='Loading {} sequences'.format(category_name)):
                seq_id = (category_name, seq_name)
                image_pathes = sorted(glob.glob(os.path.join(self.images_dir, category_name, seq_name, '*' + self.img_ext)))
                seq_images = list()
                for image_path in image_pathes:
                    if not os.path.exists(image_path):
                        raise IOError('File not found: {}'.format(image_path))
                    image = cv.imread(image_path)  # HWC BGR image
                    if self.should_downscale_images and image.shape[0] > self.downscale_height:
                        downscale_factor = self.downscale_height / float(image.shape[0])
                        image = cv.resize(image, None,
                                          fx=downscale_factor,
                                          fy=downscale_factor, interpolation=cv.INTER_LINEAR)
                    seq_images.append(image)
                self.sequences[seq_id] = seq_images
        # TODO: balance categories
        self.index2seqid = self.sequences.keys()

    def __len__(self):
        return len(self.sequences)

    def random_crop(self, image, crop_size):
        """
        Randomly crop the image.
        """

        x = y = 0
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size[:2]

        shift_x = self.rs.randint(0, w - crop_w + 1)
        shift_y = self.rs.randint(0, h - crop_h + 1)

        x += shift_x
        y += shift_y

        image = image[y:y + crop_h, x:x + crop_w]
        return image

    def get_example(self, i):
        """
        Get clip
        """
        seq_id = self.index2seqid[i]
        images = self.sequences[seq_id]

        if self.random_clip_subsample_type is None:
            period = len(images) / self.num_frames_in_clip
            first_frame = 0
        elif self.random_clip_subsample_type == RandomClipSubsampleType.CONSECUTIVE:
            period = 1
            first_frame = self.rs.randint(len(images) - self.num_frames_in_clip + 1)
        else:
            raise NotImplementedError

        fliplr = self.rs.randint(self.fliplr + 1)

        clip = list()
        for pos in xrange(first_frame, len(images), period):
            frame = np.array(images[pos])
            if fliplr:
                frame = cv.flip(frame, 1)
            if frame.shape[:2] != self.image_size:
                frame = cv.resize(frame, dsize=self.image_size, interpolation=cv.INTER_LANCZOS4)
            if frame.shape[:2] != self.crop_size:
                frame = self.random_crop(frame, self.crop_size)
            clip.append(frame)

        clip = np.asarray(clip, dtype=np.float32)  # NxHxWxC BGR
        assert clip.shape == (self.num_frames_in_clip,) + self.crop_size + (3,)
        return clip
