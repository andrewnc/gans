import sys

import spacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import pickle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import fastai.vision as fv


BATCH_SIZE = 64

# close the file
n_epochs = 200
batch_size = BATCH_SIZE
lr = 0.00005
b1 = 0.5
b2 = 0.999
n_cpu=8
latent_dim = 100
bert_dim = 768
img_size = 64
n_critic=5
clip_value=0.01
channels = 3
sample_interval = 400

os.makedirs("images", exist_ok=True)


img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + bert_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

with open("df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("test_df.pkl", "rb") as f:
    test_df = pickle.load(f)

with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)
data = fv.ImageDataBunch.from_df("train",train_df, valid_pct=0, size=img_size, bs=batch_size)
test_data = fv.ImageDataBunch.from_df("test",test_df, valid_pct=0, size=img_size, bs=batch_size)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(n_epochs):

    for i, (imgs, text_inds) in enumerate(data.train_dl):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        texts = Tensor([mapping[ind.item()][1] for ind in text_inds])

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        z = torch.cat((z, texts), 1)

        # Generate a batch of images
        fake_imgs = generator(z)
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, batches_done % len(data.train_dl), len(data.train_dl), loss_D.item(), loss_G.item())
            )

        if batches_done % sample_interval == 0:
            with open("images/descriptions.txt", "a+") as f:
                f.write("\n{}----\n".format(batches_done) + "\n".join([mapping[ind.item()][0] for ind in text_inds[:25]]))
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            with open("wgan_generator.pkl", "wb") as f:
                pickle.dump(generator, f)
            with open("wgan_discriminator.pkl", "wb") as f:
                pickle.dump(discriminator, f)
        batches_done += 1