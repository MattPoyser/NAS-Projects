##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from .DownsampledImageNet import ImageNet16
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
sys.path.insert(0, "/home2/lgfm95/hem/perceptual")
sys.path.insert(0, "C:\\Users\\Matt\\Documents\\PhD\\x11\\HEM\\perceptual")
sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from dataloader_classification import DynamicDataset
from subloader import SubDataset

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'mnist': 10,
                 'fashion': 10,
                 'imagenet': 1000,
                 'imagenet-1k-s': 1000,
                 'imagenet-1k': 1000,
                 'ImageNet16': 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

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


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_datasets(name, root, config):
    cutout = config.cutout_length
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name == 'tiered':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name == 'imagenet-1k' or name == 'imagenet-100' or name == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name == 'mnist':
        mean = [0.13066051707548254]
        std = [0.30810780244715075]
    elif name == 'fashion':
        mean = [0.28604063146254594]
        std = [0.35302426207299326]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100' or name == 'mnist' or name == 'fashion':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
        if name == "mnist" or name == "fashion":
            test_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
            xshape = (1, 1, 32, 32)
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 16, 16)
    elif name == 'tiered':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('imagenet-1k')or name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k' or name == 'imagenet':
            xlists = [transforms.RandomResizedCrop(224)]
            xlists.append(
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2))
            xlists.append(Lighting(0.1))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
        else:
            raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.RandomHorizontalFlip(p=0.5))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        xshape = (1, 3, 224, 224)
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    pretrain_resume = "/home2/lgfm95/hem/perceptual/good.pth.tar"
    grayscale = False
    is_detection = False
    convert_to_paths = False
    convert_to_lbl_paths = False
    isize = 64
    nz = 8
    aisize = 3
    if name == 'cifar10':
        dset_cls = dset.CIFAR10
        # train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
        # test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=True)
        dynamic_name = "cifar10"
        n_classes = 10
        nz = 32
        auto_resume = "/home2/lgfm95/hem/perceptual/tripletCifarMseKGood.pth.tar"
        # assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(root, train=False, transform=test_transform, download=True)
        dset_cls = dset.CIFAR100
        dynamic_name = "cifar100"
        n_classes = 100
        nz = 32
        auto_resume = "/home2/lgfm95/hem/perceptual/tripletCifarMseKGood.pth.tar"
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k') or name == 'imagenet':
        # train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        # test_data = dset.ImageFolder(osp.join(root, 'val'), test_transform)
        dynamic_name = "imagenet"
        n_classes = 1000
        if config.ncc:
            auto_resume = "/home2/lgfm95/hem/perceptual/ganPercImagenetGood.pth.tar"
        else:
            auto_resume = "/hdd/PhD/hem/perceptual/ganPercImagenetGood.pth.tar"
        isize = 256
        convert_to_paths = True
        # assert len(train_data) == 1281167 and len(
        #     test_data) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data),
        #                                                                                     len(test_data), 1281167,
        #                                                                                     50000)

    elif name == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
        dynamic_name = "mnist"
        grayscale = True
        aisize = 1
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercMnistGood.pth.tar"
    elif name == 'fashion':
        dset_cls = dset.FashionMNIST
        n_classes = 10
        dynamic_name = "fashion"
        grayscale = True
        aisize = 1
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercFashionGood.pth.tar"
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    normalize = transforms.Normalize(
        mean=[0.13066051707548254],
        std=[0.30810780244715075])
    perc_transforms = transforms.Compose([
        transforms.RandomResizedCrop(isize),
        transforms.ToTensor(),
        normalize,
    ])
    if config.isbad:
        auto_resume = "badpath"

    if config.dynamic:
        # print(perc_transforms)
        train_data = DynamicDataset(
            perc_transforms=perc_transforms,
            pretrain_resume=pretrain_resume,
            image_transforms=train_transform,
            val_transforms=test_transform,
            val=False,
            dataset_name=dynamic_name,
            auto_resume=auto_resume,
            hardness=config.hardness,
            isize=isize,
            nz=nz,
            aisize=aisize,
            grayscale=grayscale,
            isTsne=True,
            tree=config.isTree,
            subset_size=config.subset_size,
            is_csv=config.is_csv,
            is_detection=is_detection,
            is_concat=False,
            seed=1337) # TODO
        # is_csv=False)
        if name == "imagenet":
            test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name,
                                   subset_size=10000)
        else:
            test_data = dset_cls(root=root, train=False, download=False, transform=test_transform)
    else:
        if config.vanilla:
            if name == "imagenet":
                # subset_size = 10000
                # train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                #                         dataset_name=dynamic_name, subset_size=subset_size)
                # test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)
                train_data = dset.ImageFolder(
                    os.path.join(root, "train"),
                    train_transform)
                test_data = dset.ImageFolder(
                    os.path.join(root, "val"),
                    test_transform)
            else:
                train_data = dset_cls(root=root, train=True, download=False, transform=train_transform)
                test_data = dset_cls(root=root, train=False, download=False, transform=test_transform)
        else:
            if name == "imagenet":
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                                        dataset_name=dynamic_name, subset_size=config.subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=config.subset_size)
            else:
                subset_size = config.subset_size
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False, dataset_name=dynamic_name, subset_size=subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_datasets_augment(name, root, cutout, kd=False):
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k') or name == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    elif name == 'mnist':
        mean = [0.13066051707548254]
        std = [0.30810780244715075]
    elif name == 'fashion':
        mean = [0.28604063146254594]
        std = [0.35302426207299326]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100' or name == "mnist" or name == "fashion":
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
        if name == "mnist" or name == "fashion":
            test_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
            xshape = (1, 1, 32, 32)
            if kd: #no longer needed as we have retrained to have proper grayscale input channel
                convertrgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                newlists = [elem for elem in lists]
                newlists.append(convertrgb)
                train_transform = transforms.Compose(newlists)
                test_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std), convertrgb])
                xshape = (1, 3, 32, 32) # change back to 3 channel input
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 16, 16)
    elif name == 'tiered':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('imagenet-1k') or name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k' or name == 'imagenet':
            xlists = [transforms.RandomResizedCrop(224)]
            xlists.append(
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2))
            xlists.append(Lighting(0.1))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
        else:
            raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.RandomHorizontalFlip(p=0.5))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        xshape = (1, 3, 224, 224)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "mnist":
        train_data = dset.MNIST(root, train=True, transform=train_transform, download=False)
        test_data = dset.MNIST(root, train=False, transform=test_transform, download=False)
        assert len(train_data) == 60000 and len(test_data) == 10000
    elif name == "fashion":
        train_data = dset.FashionMNIST(root, train=True, transform=train_transform, download=False)
        test_data = dset.FashionMNIST(root, train=False, transform=test_transform, download=False)
        assert len(train_data) == 60000 and len(test_data) == 10000, f"{len(train_data)} / {len(test_data)}"
    elif name == 'cifar10':
        train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=False)
        test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=False)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k') or name == 'imagenet':
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data = dset.ImageFolder(osp.join(root, 'val'), test_transform)
        assert len(train_data) == 1281167 and len(
            test_data) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data),
                                                                                            len(test_data), 1281167,
                                                                                            50000)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num
# if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
