import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        use_depth_hints
        depth_hint_path
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_depth_hints,
                 depth_hint_path=None,
                 is_train=False,
                 img_ext='.jpg',
                 is_test = False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.use_depth_hints = use_depth_hints

        # assume depth hints npys are stored in data_path/depth_hints unless specified
        if depth_hint_path is None:
            self.depth_hint_path = os.path.join(self.data_path, 'depth_hints')
        else:
            self.depth_hint_path = depth_hint_path

        # self.interp = Image.ANTIALIAS
        self.interp = transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1


        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)
        self.load_depth = True
        self.is_test = is_test
        self.max = 0

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_aug" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if color_aug is None:
                if "color" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if "color_aug" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
            else:
                if "color" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    if inputs[(n, im, i)].sum() == 0:
                        inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                    else:
                        inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

                # if color_aug is not None:
                #     if inputs[(n, im, i)].sum() == 0:
                #         inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                #     else:
                #         inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                # else:
                #     if inputs[(n, im, i)].sum() == 0:
                #         inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                #     else:
                #         inputs[(n + "_aug", im, i)] = self.to_tensor(inputs[("rawcolor", im, i)])




    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps
            "depth_hint"                            for depth hint
            "depth_hint_mask"                       for mask of valid depth hints

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        if self.is_train:
            do_color_aug = self.is_train and random.random() > 0.5
        else:
            do_color_aug = True
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])
        side = None
        poses = {}

        for i in self.frame_idxs:# 0,-1,1
            try:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip,is_seaErra = True)# image name + 0/-1/1
                inputs[("color_aug", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)  # image name + 0/-1/1
            except FileNotFoundError as e:
                if i != 0:
                    # fill with dummy values
                    inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                    poses[i] = None
                    inputs[("color_aug", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                    poses[i] = None
                else:
                    raise FileNotFoundError(f'Cannot find frame - make sure your '
                                            f'--data_path is set correctly, or try adding'
                                            f' the --png flag. {e}')
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            if folder in [ 'flatiron','u_canyon', 'tiny_canyon','horse_canyon']:#'flatiron','landward_path'
                K = self.K_canyons.copy()
            else:
                K = self.K_red_sea.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            # color_aug = (lambda x: x)
            color_aug = None
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            # if inputs[("rawcolor", i, -1)]:
            #     del inputs[("rawcolor", i, -1)]
            # if inputs[("rawcolor", i, 0)]:
            #     del inputs[("rawcolor", i, 0)]
            # if inputs[("rawcolor", i, 1)]:
            #     del inputs[("rawcolor", i, 1)]
            # if inputs[("rawcolor", i, 2)]:
            #     del inputs[("rawcolor", i, 2)]
            # if inputs[("rawcolor", i, 3)]:
            #     del inputs[("rawcolor", i, 3)]



        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        if self.is_test:
             inputs["folder"] = folder
             inputs["frame_index"] = frame_index
        # if self.max < mm:
        #     self.max = mm
        #     print(self.max)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError