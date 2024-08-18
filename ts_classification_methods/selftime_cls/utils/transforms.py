import random
import torch
from utils.augmentation import *


class Raw:
    def __init__(self):
        pass

    def __call__(self, data):
        return data


class CutPiece2C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece2C(data, self.sigma)


class CutPiece3C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece3C(data, self.sigma)


class CutPiece4C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece4C(data, self.sigma)


class CutPiece5C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece5C(data, self.sigma)


class CutPiece6C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece6C(data, self.sigma)


class CutPiece7C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece7C(data, self.sigma)


class CutPiece8C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        return cut_piece8C(data, self.sigma)


class Jitter:
    def __init__(self, sigma, p):
        self.sigma = sigma
        self.p = p

    def __call__(self, data):
        # print('### Jitter')

        if random.random() < self.p:
            return self.forward(data)
        return data

    def forward(self, data):

        return jitter(data, sigma=self.sigma)


class Scaling:
    def __init__(self, sigma, p):
        self.sigma = sigma
        self.p = p

    def __call__(self, data):
        # print('### Scaling')

        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return scaling_s(data, sigma=self.sigma)


class Cutout:
    def __init__(self, sigma, p):
        self.sigma = sigma
        self.p = p

    def __call__(self, data):
        # print('### Cutout')

        if random.random() < self.p:
            return self.forward(data)
        return data

    def forward(self, data):
        return cutout(data, self.sigma)


class MagnitudeWrap:
    def __init__(self, sigma, knot, p):
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, data):
        # print('### MagnitudeWrap')

        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return magnitude_warp_s(data, sigma=self.sigma, knot=self.knot)


class TimeWarp:
    def __init__(self, sigma, knot, p):
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return time_warp_s(data, sigma=self.sigma, knot=self.knot)


class WindowSlice:
    def __init__(self, reduce_ratio, p):
        self.reduce_ratio = reduce_ratio
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return window_slice_s(data, reduce_ratio=self.reduce_ratio)


class WindowWarp:
    def __init__(self, window_ratio, scales, p):
        self.window_ratio = window_ratio
        self.scales = scales
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return window_warp_s(data, window_ratio=self.window_ratio, scales=self.scales)


class ToTensor:
    '''
    Attributes
    ----------
    basic : convert numpy to PyTorch tensor

    Methods
    -------
    forward(img=input_image)
        Convert HWC OpenCV image into CHW PyTorch Tensor
    '''
    def __init__(self, basic=False):
        self.basic = basic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv/numpy image

        Returns
        -------
        Torch tensor
            BGR -> RGB, [0, 255] -> [0, 1]
        '''
        ret = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        return ret


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)

        return img
