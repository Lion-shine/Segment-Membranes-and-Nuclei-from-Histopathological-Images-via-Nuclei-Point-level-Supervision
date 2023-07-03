from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import deepdish as dd
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize
import torch
from pydaily import filesystem
import h5py
import time
def read_image(data_dir):
    png_list = filesystem.find_ext_files(data_dir, "h5")
    return png_list
class DataFolder(Dataset):
    def __init__(self, data_dir, resize_ratio_list = (0.8, 0.9, 1.0, 1.1, 1.2),
                 image_size = 206,smallest_size=206,  use_grey=False, is_training=True):
        self.__dict__.update(locals())
        self.data_dir = data_dir
        self.smallest_size = smallest_size
        # self.transform = transform
        self.is_training = is_training
        self.use_grey = use_grey
        self.resize_ratio_list = resize_ratio_list
        self.data_list =read_image(self.data_dir)
        self._base_trans  = transforms.Compose([
                            RandomRotation_list([0, 90, 180]),
                            transforms.RandomCrop(image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                        ])

    def transform(self, image, mask):
        '''
        :param image:  np array
        :param mask:  np array
        :return:
        '''
        image = Image.fromarray(image)
        # mask = Image.fromarray(mask[:,:,0])
        image = transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                saturation=0.3, hue=0.2)(image)
        seed = random.randint(0, 2 ** 64)
        random.seed(seed)
        image = self._base_trans(image)


        if len(mask.shape) ==3 and mask.shape[2]>1:
            mask_list = []
            for i in range(mask.shape[2]-1):
                this_mask = mask[:, :, i]
                this_mask = Image.fromarray(this_mask)
                random.seed(seed)
                this_mask = self._base_trans(this_mask)
                this_mask = TF.to_tensor(this_mask)
                mask_list.append(this_mask)
            mask = torch.stack(mask_list)
            # import pdb; pdb.set_trace()
            if len(mask.size()) == 4:
                mask = torch.squeeze(mask)
        else:
            random.seed(seed)
            mask = self._base_trans(mask)
        mask = mask/255.
        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return image, mask

    def test_transform(self, image, mask):
        image = Image.fromarray(image)
        seed = random.randint(0, 2 ** 64)
        random.seed(seed)
        image = transforms.RandomCrop(512)(image)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if len(mask.shape) ==3 and mask.shape[2]>1:
            mask_list = []
            for i in range(mask.shape[2]-1):
                this_mask = mask[:, :, i]
                this_mask = Image.fromarray(this_mask)
                random.seed(seed)
                this_mask = transforms.RandomCrop(512)(this_mask)
                this_mask = TF.to_tensor(this_mask)
                mask_list.append(this_mask)
            mask = torch.stack(mask_list)
            if len(mask.size()) == 4:
                mask = torch.squeeze(mask)
        else:
            random.seed(seed)
            mask = transforms.RandomCrop(512)(mask)
        mask = mask/255.
        return image, mask


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        while True:
            with h5py.File(self.data_list[index],'r') as cur_dict:
                img, mask = cur_dict['img'][:], cur_dict['mask'][:]
                ratio_list = self.resize_ratio_list
                resize_ratio = random.choice(ratio_list)
                resize_h, resize_w = int(img.shape[0] * resize_ratio), int(img.shape[1] * resize_ratio)
                img = cv2.resize(img, (resize_w, resize_h))

                mask = sk_resize(mask, (resize_h, resize_w))

                # 2. pad image and mask:
                ori_h, ori_w = img.shape[0:2]
                pad_h, pad_w = max(0, self.smallest_size - ori_h), max(0, self.smallest_size - ori_w)
                if pad_h > 0 or pad_w > 0:
                    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
                    if len(mask.shape) == 3:
                        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
                    else:
                        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'reflect')

                # imsave("img.png", img)
                # for i in range(mask.shape[2]):
                #     imsave("mask_{}.png".format(i), mask[:,:,i])

                # 3. data augmentation
                if self.is_training:
                    img, mask = self.transform(img, mask)  # img, mask are numpy array
                else:
                    img, mask = self.test_transform(img, mask)

                if self.use_grey is True:
                    img = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
                    img = img.unsqueeze(0)
                return [img, mask]

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

