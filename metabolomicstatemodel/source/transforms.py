import PIL
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms.functional as TF
from torchvision import transforms
import pytorch_lightning as pl
import random
from kornia.augmentation.mix_augmentation import RandomCutMix, RandomMixUp


class AbstractTransforms(pl.LightningModule):
    """
    Abstract class for riskiano transformations.

    Provides flexible interface for train and valid trainsforms.

    If sample is a tuple, transformations are always applied to the first element of the tuple.
    """

    def __init__(self, device='cuda', cut_mix=False, mix_up=False):
        super().__init__()
        self.train_transform = nn.Sequential().to(device)
        self.valid_transform = nn.Sequential().to(device)
        self.test_transfrom = nn.Sequential().to(device)
        self.cut_mix = cut_mix
        self.mix_up = mix_up

    @torch.no_grad()
    def apply_train_transform(self, batch, targets):
        if isinstance(batch, tuple) or isinstance(batch, list):
            transformed = self.train_transform(batch[0])
            return (transformed, *batch[1:])
        else:
            batch = self.train_transform(batch)

        if self.cut_mix:
            num_mix = 1
            cut_mix = RandomCutMix(*batch.shape[2:], num_mix=num_mix)
            batch, mix_labels = cut_mix(batch, targets.squeeze(dim=1))

            raw_labels, permuted_labels, lambdas = mix_labels[num_mix - 1, :, 0], mix_labels[num_mix - 1, :,1], \
                                                   mix_labels[num_mix - 1, :, 2]
            targets = (raw_labels * (1 - lambdas) + permuted_labels * lambdas).unsqueeze(dim=1)

        elif self.mix_up:
            mixup = RandomMixUp()
            batch, mix_labels = mixup(batch, targets.squeeze(dim=1))
            raw_labels, permuted_labels, lambdas = mix_labels[:, 0], mix_labels[:,1], mix_labels[:, 2]
            targets = (raw_labels * (1 - lambdas) + permuted_labels * lambdas).unsqueeze(dim=1)

        return batch, targets


    @torch.no_grad()
    def apply_valid_transform(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            transformed = self.valid_transform(batch[0])
            return (transformed, *batch[1:])
        else:
            transformed = self.valid_transform(batch)
            return transformed

    @torch.no_grad()
    def forward(self, batch):
        raise NotImplementedError('use the explicit transformation methods')
        transformed = self.transforms(batch)
        return transformed


class TransformsFromList(AbstractTransforms):
    """
    Wrapper to instatiate transformation object from lists.
    """

    def __init__(self, device='cuda', valid_transforms_list=[], train_transforms_list=[], cut_mix=False, mix_up=False):
        super().__init__(cut_mix=cut_mix, mix_up=mix_up)
        self.train_transform = nn.Sequential(*train_transforms_list).to(device)
        self.valid_transform = nn.Sequential(*valid_transforms_list).to(device)


class AdaptiveRandomCropTransform(nn.Module):
    def __init__(self, ratio, out_size, interpolation=PIL.Image.BILINEAR):
        super().__init__()
        self.ratio = ratio
        self.out_size = out_size
        self.interpolation = interpolation

    def forward(self, sample):
        input_size = min(sample.size)
        crop_size = int(self.ratio*input_size)
        if crop_size < self.out_size:
            crop_size = tv.transforms.transforms._setup_size(self.out_size,
                                                             error_msg="Please provide only two dimensions (h, w) for size.")
            i, j, h, w = transforms.RandomCrop.get_params(sample, crop_size)
            return TF.crop(sample, i, j, h, w)
        else:
            crop_size = tv.transforms.transforms._setup_size(crop_size,
                                                             error_msg="Please provide only two dimensions (h, w) for size.")
            i, j, h, w = transforms.RandomCrop.get_params(sample, crop_size)
            cropped = TF.crop(sample, i, j, h, w)
        return TF.resize(cropped, self.out_size, self.interpolation)


class RandomChoice(nn.Module):
    """
    Apply a randomly chosen transformation from a list.
    """
    def __init__(self, funcs):
        super().__init__()
        self.funcs = funcs

    def forward(self, x):
        func = random.choice(self.funcs)
        return func(x)


class RandomApply(nn.Module):
    """
    Randomly apply augmentation with probability p.
    """
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)