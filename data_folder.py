from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import deepdish as dd
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import torch.nn.functional as func
import torch
from pydaily import filesystem
import h5py
import time
def read_image(data_dir):
    png_list = filesystem.find_ext_files(data_dir, "h5")
    return png_list
class DataFolder(Dataset):
    def __init__(self, data_dir, resize_ratio_list = (0.8, 0.9, 1.0, 1.1, 1.2),
                 image_size = 206,smallest_size=206, augmentation_prob=0.4, is_training=True):
        self.__dict__.update(locals())
        self.data_dir = data_dir
        self.smallest_size = smallest_size
        # self.transform = transform
        self.is_training = is_training
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        self.resize_ratio_list = resize_ratio_list
        self.data_list =read_image(self.data_dir)
        self.image_size=image_size
        self._base_trans  = transforms.Compose([
                            transforms.RandomCrop(image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
                        ])

    def to_one_hot(self,mask, n_class):
        """
        Transform a mask to one hot
        change a mask to n * h* w   n is the class
        Args:
            mask:
            n_class: number of class for segmentation
        Returns:
            y_one_hot: one hot mask
        """
        y_one_hot = torch.zeros((mask.shape[1], mask.shape[2],n_class))
        y_one_hot = y_one_hot.scatter(0, mask, 1).long()
        return y_one_hot

    def transform(self, image, mask,point_dist):
        '''
        :param image:  np array
        :param mask:  np array
        :return:
        '''
        image = Image.fromarray(image)
        point_dist = Image.fromarray(point_dist)
        GT_list=[]
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size))
        image = TF.crop(image, i, j, h, w)
        point_dist = TF.crop(point_dist, i, j, h, w)
        for ind in range(mask.shape[2]):
            GT=mask[:,:,ind]
            GT=np.repeat(GT[:,:,np.newaxis],3,axis=-1).astype(np.uint8)
            GT = Image.fromarray(GT)
            GT = TF.crop(GT, i, j, h, w)
            GT_list.append(GT)
        Transform = []
        p_transform = random.random()
        if p_transform <= self.augmentation_prob:
            if random.random() < 0.5:
                image = F.hflip(image)
                point_dist = F.hflip(point_dist)
                for ind,GT in enumerate(GT_list):
                    GT_list[ind] = F.hflip(GT)
            if random.random() < 0.5:
                image = F.vflip(image)
                point_dist = F.vflip(point_dist)
                for ind, GT in enumerate(GT_list):
                    GT_list[ind] = F.vflip(GT)
        Transform.append(transforms.ToTensor())
        Transform = transforms.Compose(Transform)
        image = Transform(image)
        point_dist = Transform(point_dist)
        # point_dist = point_dist * 255
        GT_tensor=[]
        for ind, GT in enumerate(GT_list):
            GT=Transform(GT)[0].unsqueeze(0)
            GT_tensor.append(GT)

        GT=torch.cat(GT_tensor,dim=0)
        Norm_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        return [image, GT,point_dist]


    def test_transform(self, image, mask,point_dist):
        image = Image.fromarray(image)
        point_dist = Image.fromarray(point_dist)
        GT_list = []
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size))
        image = TF.crop(image, i, j, h, w)
        point_dist = TF.crop(point_dist, i, j, h, w)
        for ind in range(mask.shape[2]):
            GT = mask[:, :, ind]
            GT = np.repeat(GT[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)
            GT = Image.fromarray(GT)
            GT = TF.crop(GT, i, j, h, w)
            GT_list.append(GT)
        Transform = []
        Transform.append(transforms.ToTensor())
        Transform = transforms.Compose(Transform)
        image = Transform(image)
        point_dist = Transform(point_dist)
        GT_tensor = []
        for ind, GT in enumerate(GT_list):
            GT = Transform(GT)[0].unsqueeze(0)
            GT_tensor.append(GT)
        GT = torch.cat(GT_tensor, dim=0)
        Norm_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        return [image, GT,point_dist]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        while True:
            with h5py.File(self.data_list[index],'r') as cur_dict:
                img, mask,point_dist = cur_dict['img'][:], cur_dict['mask'][:],cur_dict['point_dist'][:]
                # point_dist=point_dist/255.0
                if self.is_training:
                    img, mask,point_dist = self.transform(img, mask,point_dist)  # img, mask are numpy array
                    return [img, mask,point_dist]
                else:
                    img, mask,point_dist = self.test_transform(img, mask,point_dist)
                    return [img, mask,point_dist]

class RandomRotation_list(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    def __init__(self, degrees=None, resample=False, expand=False, center=None):
        if isinstance(degrees, list):
            self.degrees = degrees
        else:
            raise ValueError("degree must be list.")
        self.resample = resample
        self.expand = expand
        self.center = center
    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)  # random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.degrees)
        return TF.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

