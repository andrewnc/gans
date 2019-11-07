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
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu=8
latent_dim = 100
bert_dim = 768
img_size = 64
channels = 3
sample_interval = 400

cuda = True if torch.cuda.is_available() else False

# we can probably comment this out in the end actually
# print("loading bert langauge model")
# nlp = spacy.load("en_pytt_bertbaseuncased_lg")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# DCGAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + bert_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

with open("df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("test_df.pkl", "rb") as f:
    test_df = pickle.load(f)

with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# now, we need to add the data pairing.

# oh wait, ok. it is going to be a dict mapping index to [words, vectors]
# so I will still have to construct the tensor, but I won't have to use #nlp

data = fv.ImageDataBunch.from_df("train",train_df, valid_pct=0, size=img_size, bs=batch_size)
test_data = fv.ImageDataBunch.from_df("test",test_df, valid_pct=0, size=img_size, bs=batch_size)
# for i, (imgs, _) in enumerate(data.train_dl):

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

images = data.x
descriptions = data.y
os.makedirs("images", exist_ok=True)
for epoch in range(n_epochs):
    for i, (imgs, text_inds) in enumerate(data.train_dl):
        # this is the slowest part.... 
        # texts = Tensor([nlp(data.y.get(ind).obj).vector for ind in text_inds])
        texts = Tensor([mapping[ind.item()][1] for ind in text_inds])

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        # we need to concatenate the text properly
        z = torch.cat((z, texts), 1)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(data.train_dl), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(data.train_dl) + i
        if batches_done % sample_interval == 0:
            with open("images/descriptions.txt", "a+") as f:
                f.write("\n{}----\n".format(batches_done) + "\n".join([mapping[ind.item()][0] for ind in text_inds[:25]]))
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    with open("generator.pkl", "wb") as f:
        pickle.dump(generator, f)
    with open("discriminator.pkl", "wb") as f:
        pickle.dump(discriminator, f)
# is_using_gpu = spacy.prefer_gpu()
# if is_using_gpu:
#     torch.set_default_tensor_type("torch.cuda.FloatTensor")

# print("loading bert langauge model")
# nlp = spacy.load("en_pytt_bertbaseuncased_lg")


# print("ready to rock")
# doc = nlp("Here is some text to encode.")
# doc1 = nlp("There is some text to encode.")

# print(doc.similarity(doc1))
