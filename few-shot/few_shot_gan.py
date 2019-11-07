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
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch

import fastai.vision as fv


BATCH_SIZE = 128

# close the file
n_epochs = 10
batch_size = BATCH_SIZE
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu=8
latent_dim = 100
concat_dim = 0
img_size = 64
channels = 1
n_critic = 5
sample_interval = 100

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
# We need to be able to update just the last "head" layer on both the generator and discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim+concat_dim, 128 * self.init_size ** 2))

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
        self.adv_layer = nn.Sequential(nn.Linear(128  * (ds_size ** 2), 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

train_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
test_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

train_loader_fashion = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
test_loader_fashion = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# here we need to construct the test, train, S, sets for our meta learning process
# potentially we will just do this by moving a set of folders into a "train" or "meta-learn" folder
# data = fv.ImageDataBunch.from_folder("Data/zucchini/", valid_pct=.3, size=img_size, bs=batch_size)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
idn = torch.eye(9).cuda()

def train_standard_gan(generator, discriminator, data, train_epochs, name=""):
    """train generator and discriminator first time"""
    output_dir = "{}_few_shot_images".format(name)
    os.makedirs(output_dir, exist_ok=True)
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


    for epoch in range(train_epochs):

        # here we need to construct an inner and outer loop
        for i, (imgs, target) in enumerate(data):
            imgs = imgs.cuda()
            target = target.cuda()

            # Adversarial ground truths
            # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            valid = torch.ones((imgs.shape[0], 1)).cuda()
            fake = torch.zeros((imgs.shape[0],1)).cuda()

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            gen_imgs = generator(z)

            fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()


            # Generate a batch of images
            # z = torch.cat((z, idn[target - 1]), 1)
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, train_epochs, i, len(data), d_loss.item(), g_loss.item())
            )


            batches_done = epoch * len(data) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], output_dir + "/%d.png" % batches_done, nrow=5, normalize=True)
        with open("{}_generator.pkl".format(name), "wb") as f:
            pickle.dump(generator, f)
        with open("{}_discriminator.pkl".format(name), "wb") as f:
            pickle.dump(discriminator, f)
    return generator, discriminator



# Initialize generator and discriminator
org_generator = Generator()
org_discriminator = Discriminator()

generator = org_generator
discriminator = org_discriminator

generator, discriminator = train_standard_gan(org_generator, org_discriminator, train_loader_fashion, n_epochs,name="fashion_initial")

# reset the top and freeze most of the generator
generator.l1 = nn.Linear(latent_dim+concat_dim, 128 * (img_size // 4) ** 2)
for i in range(len(generator.conv_blocks)):
    generator.conv_blocks[i].requires_grad = False

# reset the head
# generator.conv_blocks[9] = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# generator.conv_blocks[0] = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

for i in range(15):
    discriminator.model[i].requires_grad = False

# reset the head
discriminator.adv_layer[0] = nn.Linear(in_features=2048, out_features=1, bias=True)

f_generator, f_discriminator = train_standard_gan(generator, discriminator, train_loader_mnist, 1,name="mnist_transfer")


def train_wasgan_gp(generator, discriminator, data, train_epochs, name=""):
    output_dir = "{}_few_shot_images".format(name)
    os.makedirs(output_dir, exist_ok=True)
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(train_epochs):

        # here we need to construct an inner and outer loop
        for i, (real_imgs, target) in enumerate(data):
            real_imgs = real_imgs.cuda()
            target = target.cuda()
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_dim))))
            # z = torch.cat((z, idn[target - 1]), 1)

            # Generate a batch of images
            fake_imgs = generator(z.view(-1,latent_dim,1,1))
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

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
                % (epoch, n_epochs, i, len(data), d_loss.item(), loss_G.item())
            )
            batches_done = epoch * len(data) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], output_dir + "/%d.png" % batches_done, nrow=5, normalize=True)
    return generator, discriminator