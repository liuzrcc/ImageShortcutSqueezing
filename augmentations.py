import torch
import numpy as np
import random
from cv2 import resize
from torch.nn import functional as F
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO



def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(torch.utils.data.Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class MixUp(torch.utils.data.Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

class CutMixCrossEntropyLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)





cifar10 = datasets.CIFAR10(root="~/data/")
img = cifar10[100][0]

class channel_mean(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        img = np.mean(img, axis=2)
        res[:, :, 0] = img
        res[:, :, 1] = img
        res[:, :, 2] = img
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res
        
class channel_single(object):
    def __init__(self, channel):
        self.channel = channel
        self.search_dict ={'R':0, 'G':1, 'B':2}

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        img = img[:, :, self.search_dict[self.channel]]
        res[:, :, 0] = img
        res[:, :, 1] = img
        res[:, :, 2] = img
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res

class color_shift(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        res[:, :, 0] = img[:, :, 1]
        res[:, :, 1] = img[:, :, 2]
        res[:, :, 2] = img[:, :, 0]
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res

class one_channel_0(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        res[:, :, 0] = img[:, :, 0]
        res[:, :, 1] = img[:, :, 1]
        res[:, :, 2] = img[:, :, 1]
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res

class fixed_len_add(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        # print(img)
        res = np.zeros_like(img)
        res[:, :, 0] = img[:, :, 0]
        res[:, :, 1] = img[:, :, 1]
        res[:, :, 2] = img[:, :, 1]
        # res[:, :, 2] = np.clip(img[:, :, 1] + 10, 0, 255)
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res


class ratio_shrink(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        res[:, :, 0] = img[:, :, 0] / 4
        res[:, :, 1] = img[:, :, 1] / 4
        res[:, :, 2] = img[:, :, 2] / 4
        res = np.clip(res, 0, 255)
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res

class hue_limit(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        res = np.zeros_like(img)
        res[:, :, 0] = img[:, :, 0] / 4
        res[:, :, 1] = img[:, :, 1] / 4
        res[:, :, 2] = img[:, :, 2] / 4
        res = np.clip(res, 0, 255)
        Image.fromarray(res.astype('uint8')).convert('RGB')
        return res



class MeanFilter(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, image):
        # Convert image from PIL Image to numpy array
        image = np.array(image)

        # Apply mean filter to image
        image = cv2.blur(image, (self.kernel_size, self.kernel_size))

        # Convert image back to PIL Image
        image = Image.fromarray(image)

        return image

# Add the MeanFilter transform to the transforms module

class MedianFilter(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, image):
        # Convert image from PIL Image to numpy array
        image = np.array(image)

        # Apply median filter to image
        image = cv2.medianBlur(image, self.kernel_size)

        # Convert image back to PIL Image
        image = Image.fromarray(image)

        return image




def aug_train(jpeg, grayscale, bdr, TrainAUG, low_pass):

    transform_train = transforms.Compose([])
     
    def JPEGcompression(image, jpeg=jpeg):
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=jpeg, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    if bdr is not None:
        transform_train.transforms.append(transforms.RandomPosterize(bits=bdr, p=1))

    if grayscale:
        transform_train.transforms.append(transforms.Grayscale(3))

    if jpeg is not None:
        transform_train.transforms.append(transforms.Lambda(JPEGcompression))
    
    if 'gaussian_f' in low_pass:
        transform_train.transforms.append(transforms.GaussianBlur(3, sigma=0.1))



    transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
    transform_train.transforms.append(transforms.RandomHorizontalFlip())
    transform_train.transforms.append(transforms.ToTensor())


    if 'median_f' in low_pass:
        transform_train.transforms.append(MedianFilter())

    if 'mean_f' in low_pass:
        transform_train.transforms.append(MeanFilter())

    if 'cutout' in TrainAUG:
        transform_train.transforms.append(Cutout(16))

    return transform_train

    
def aug_test(ISS_both_train_test, jpeg, grayscale, bdr):
    transform_test = transforms.Compose([])

    def JPEGcompression(image, jpeg=jpeg):
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=jpeg, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    if ISS_both_train_test:

        if grayscale:
            transform_test.transforms.append(transforms.Grayscale(3))

        if bdr:
            transform_test.transforms.append(transforms.RandomPosterize(bits=bdr, p=1))

        if jpeg:
            transform_test.transforms.append(transforms.Lambda(JPEGcompression))

    transform_test.transforms.append(transforms.ToTensor())

    return transform_test






# Add the MedianFilter transform to the transforms module

if __name__ == "__main__":
    t = transforms.Compose([transforms.Resize(4), 
    color_shift(),
    transforms.ToTensor(),
    ])
    
    # plt.imshow(np.array(img))
    # plt.imshow(t(img).numpy().transpose((1, 2, 0)))
    # print(t(img))
    plt.imshow(np.array(ratio_shrink()(img)))
    plt.show()
