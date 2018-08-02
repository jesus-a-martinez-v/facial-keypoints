import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    """
    Face Landmarks dataset.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[index, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[index, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Tranforms

class Normalize(object):
    """
    Convert a color image to grayscale and normalize the color range to [0,1].
    """

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {'image': image_copy,
                'keypoints': key_pts_copy}


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                new_height = self.output_size * height / width
                new_width = self.output_size
            else:
                new_height = self.output_size
                new_width = self.output_size * width / height
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        rescaled_image = cv2.resize(image, (new_width, new_height))

        # scale the pts, too
        key_pts = key_pts * [new_width / width, new_height / height]

        return {'image': rescaled_image,
                'keypoints': key_pts}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        key_pts = key_pts - [left, top]  # Broadcasting

        return {'image': image,
                'keypoints': key_pts}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        # if image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: Height x Width x Channels
        # torch image: Channels X Height X Width
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
