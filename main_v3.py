import os

import numpy as np

from torchvision.utils import save_image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from util.metrics import accuracy, class_accuracy
from models.models_paper_v2 import ConvNet, Generator, Discriminator
from data.datasets_v2 import load_mnist, load_svhn
from training.classif_training import classifier_train_step, cls_pretraining
import torchvision.utils as vutils

# --------------------
#     Parameters
# --------------------

# Data Loading parameters
DATA_PATH = './datasets'
IMG_SIZE = 32
BATCH_SIZE = 256 #32

VERBOSE = True

# GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes
N_CLASSES = 10

# Parameters classifier pretraining
N_EPOCHS_CLS_PRETRAINING = 25 # 25
# Learning rate classifier pretraining
LR_CLS_PRETRAINING = 3e-4

# Latent space dimension
LATENT_DIM = 100
LATENT_LABEL_DIM = 10
# Learning rate discriminator pretraining
LR_D_PRETRAINING = 1e-5 #1e-5
# Learning rate generator pretraining
LR_G_PRETRAINING = 1e-5 # 1e-5
# Parameters classifier pretraining
N_EPOCHS_GAN_PRETRAINING = 30 #30

# Learning rate classifier pretraining
LR_CLS_TRAINING = 1e-5
# Learning rate discriminator training
LR_D_TRAINING = 5e-5 # 5e-5
# Learning rate generator training
LR_G_TRAINING = 5e-5 # 5e-5
# Parameters training
N_EPOCHS_TRAINING = 100

IMGS_TO_DISPLAY_PER_CLASS = 20

# Results
RESULTS_PATH = './results'

# Pretraining paths
PRETRAINING_PATH = ''.join([RESULTS_PATH, '/pretraining'])
img_pretraining_path = ''.join([PRETRAINING_PATH, '/images'])
os.makedirs(img_pretraining_path, exist_ok=True)
models_pretraining_path = ''.join([PRETRAINING_PATH, '/models'])
os.makedirs(models_pretraining_path, exist_ok=True)

# Training paths
TRAINING_PATH = ''.join([RESULTS_PATH, '/training'])
models_training_path = ''.join([TRAINING_PATH, '/models'])
os.makedirs(models_training_path, exist_ok=True)
img_training_path = ''.join([TRAINING_PATH, '/images'])
os.makedirs(img_training_path, exist_ok=True)

# --------------------
#   Data loading
# --------------------

source_loader_train, source_loader_test = load_mnist(DATA_PATH, IMG_SIZE, BATCH_SIZE)
target_loader_train, target_loader_test = load_svhn(DATA_PATH, IMG_SIZE, BATCH_SIZE)


############################ --------------------###########################
############################   Pretraining       ###########################
############################ --------------------###########################

# Classifier pretraining on source data
classifier = ConvNet(classes=N_CLASSES).to(DEVICE)
classifier = cls_pretraining(classifier, source_loader_train, source_loader_test,
                              LR_CLS_PRETRAINING, N_EPOCHS_CLS_PRETRAINING, PRETRAINING_PATH)

if VERBOSE:
    print("Test accuracy on source :")
    class_accuracy(classifier, source_loader_test, list(range(N_CLASSES)))
    print("Test accuracy on target :")
    class_accuracy(classifier, target_loader_test, list(range(N_CLASSES)))    


# Method for storing generated images
def generate_imgs(z, fixed_label, path, epoch=0):
    gen.eval()
    fake_imgs = gen(z, fixed_label)
    fake_imgs = (fake_imgs + 1) / 2
    fake_imgs_ = vutils.make_grid(
        fake_imgs, normalize=False, nrow=IMGS_TO_DISPLAY_PER_CLASS)
    vutils.save_image(fake_imgs_, os.path.join(
        path, 'sample_' + str(epoch) + '.png'))

# GAN pretraining on target data annotated by classifier
gen = Generator(z_dim=LATENT_DIM, num_classes=N_CLASSES, label_embed_size=LATENT_LABEL_DIM).to(DEVICE)
dis = Discriminator(N_CLASSES).to(DEVICE)

g_pretrained = ''.join([RESULTS_PATH, '/pretraining', '/generator_pretrained.pth'])
d_pretrained = ''.join([RESULTS_PATH, '/pretraining', '/discriminator_pretrained.pth'])

# Define Optimizers
g_opt = optim.Adam(gen.parameters(), lr=LR_G_PRETRAINING,
                   betas=(0.5, 0.999), weight_decay=2e-5)
d_opt = optim.Adam(dis.parameters(), lr=LR_D_PRETRAINING,
                   betas=(0.5, 0.999), weight_decay=2e-5)

# Loss functions
loss_fn = nn.BCELoss()

loaded_gen, loaded_dis = False, False

if os.path.isfile(g_pretrained):
    gen.load_state_dict(torch.load(g_pretrained))
    print('loaded existing generator')
    loaded_gen = True

if os.path.isfile(d_pretrained):
    dis.load_state_dict(torch.load(d_pretrained))
    print('loaded existing discriminator')
    loaded_dis = True

if not(loaded_gen and loaded_dis):
    print('Starting Pre-training GAN')

    # Fix images for viz
    fixed_z = torch.randn(IMGS_TO_DISPLAY_PER_CLASS*N_CLASSES, LATENT_DIM)
    fixed_label = torch.arange(0, N_CLASSES)
    fixed_label = torch.repeat_interleave(fixed_label, IMGS_TO_DISPLAY_PER_CLASS)

    # Labels
    real_label = torch.ones(BATCH_SIZE)
    fake_label = torch.zeros(BATCH_SIZE)

    # GPU Compatibility
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        gen, dis = gen.cuda(), dis.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        fixed_z, fixed_label = fixed_z.cuda(), fixed_label.cuda()

    total_iters = 0
    max_iter = len(target_loader_train)

    # GAN pre-Training
    for epoch in range(N_EPOCHS_GAN_PRETRAINING):
        gen.train()
        dis.train()

        for i, data in enumerate(target_loader_train):

            total_iters += 1

            # Loading data
            x_real, _ = data
            z_fake = torch.randn(BATCH_SIZE, LATENT_DIM)

            if is_cuda:
                x_real = x_real.cuda()
                z_fake = z_fake.cuda()

            # Generate fake data
            _, x_label = torch.max(classifier(x_real), dim=1)
            x_fake = gen(z_fake, x_label)

            # Train Discriminator
            fake_out = dis(x_fake.detach(), x_label)
            real_out = dis(x_real.detach(), x_label)
            d_loss = (loss_fn(fake_out, fake_label) +
                    loss_fn(real_out, real_label)) / 2

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train Generator
            fake_out = dis(x_fake, x_label)
            g_loss = loss_fn(fake_out, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i % 50 == 0:
                print("Epoch: " + str(epoch + 1) + "/" + str(N_EPOCHS_GAN_PRETRAINING)
                    + "\titer: " + str(i) + "/" + str(max_iter)
                    + "\ttotal_iters: " + str(total_iters)
                    + "\td_loss:" + str(round(d_loss.item(), 4))
                    + "\tg_loss:" + str(round(g_loss.item(), 4))
                    )

        if (epoch + 1) % 5 == 0:
            generate_imgs(fixed_z, fixed_label, img_pretraining_path, epoch=epoch + 1)

    generate_imgs(fixed_z, fixed_label, img_pretraining_path)
    torch.save(gen.state_dict(), g_pretrained)
    torch.save(dis.state_dict(), d_pretrained)


############################ --------------------###########################
############################      Training       ###########################
############################ --------------------###########################

# Classifier loss and optimizer
cls_criterion = nn.CrossEntropyLoss().to(DEVICE)
cls_optimizer = optim.Adam(classifier.parameters(), lr=LR_CLS_TRAINING)

# GAN loss and optimizers
loss_fn = nn.BCELoss()
g_opt = optim.Adam(gen.parameters(), lr=LR_G_TRAINING,
                   betas=(0.5, 0.999), weight_decay=2e-5)
d_opt = optim.Adam(dis.parameters(), lr=LR_D_TRAINING,
                   betas=(0.5, 0.999), weight_decay=2e-5)

# Training

cls_trained = ''.join([RESULTS_PATH, '/training', '/classifier_trained.pth'])
g_trained = ''.join([RESULTS_PATH, '/training', '/generator_trained.pth'])
d_trained = ''.join([RESULTS_PATH, '/training', '/discriminator_trained.pth'])

loaded_cls, loaded_gen, loaded_dis = False, False, False

if os.path.isfile(cls_trained):
    gen.load_state_dict(torch.load(cls_trained))
    print('loaded existing classifier')
    loaded_cls = True

if os.path.isfile(g_trained):
    gen.load_state_dict(torch.load(g_trained))
    print('loaded existing generator')
    loaded_gen = True

if os.path.isfile(d_trained):
    dis.load_state_dict(torch.load(d_trained))
    print('loaded existing discriminator')
    loaded_dis = True

if not(loaded_gen and loaded_dis and loaded_cls):

    # Fix images for viz
    fixed_z = torch.randn(IMGS_TO_DISPLAY_PER_CLASS*N_CLASSES, LATENT_DIM)
    fixed_label = torch.arange(0, N_CLASSES)
    fixed_label = torch.repeat_interleave(fixed_label, IMGS_TO_DISPLAY_PER_CLASS)

    # Labels
    real_label = torch.ones(BATCH_SIZE, dtype=torch.float, device=DEVICE)
    fake_label = torch.zeros(BATCH_SIZE, dtype=torch.float, device=DEVICE)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        gen, dis = gen.cuda(), dis.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        fixed_z, fixed_label = fixed_z.cuda(), fixed_label.cuda()

    total_iters = 0
    max_iter = len(target_loader_train)

    print('Starting Training : \n')
    for epoch in range(N_EPOCHS_TRAINING):
        print(f'Starting epoch {epoch +1}/{N_EPOCHS_TRAINING}...', end=' ')

        for i, (images, _) in enumerate(target_loader_train):

            total_iters += 1

            gen.train()
            dis.train()

            # Step 3
            # Sample latent space and random labels
            z_fake = torch.randn(BATCH_SIZE, LATENT_DIM)
            fake_label_cls = torch.randint(0, N_CLASSES, (BATCH_SIZE,), dtype=torch.long, device=DEVICE)

            if is_cuda:
                z_fake = z_fake.cuda()
                fake_label_cls = fake_label_cls.cuda()


            # Step 4
            # Generate fake images
            x_fake = gen(z_fake, fake_label_cls)

            # Step 5
            # Update classifier
            c_loss = classifier_train_step(classifier, x_fake, cls_optimizer, cls_criterion, fake_label_cls)

            # Step 6
            # Sample real images from target data
            z_fake = torch.randn(BATCH_SIZE, LATENT_DIM)
            if is_cuda:
                x_real = images.cuda()
                z_fake = z_fake.cuda()

            # Step 7
            # Infer labels using classifier
            _, x_label = torch.max(classifier(x_real), dim=1)

            # Step 8
            # Update discriminator
            x_fake = gen(z_fake, x_label)
            fake_out = dis(x_fake.detach(), x_label)
            real_out = dis(x_real.detach(), x_label)
            d_loss = (loss_fn(fake_out, fake_label) +
                    loss_fn(real_out, real_label)) / 2

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Step 9
            # Update Generator
            fake_out = dis(x_fake, x_label)
            g_loss = loss_fn(fake_out, real_label)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
            
            if i % 50 == 0:
                print("Epoch: " + str(epoch + 1) + "/" + str(N_EPOCHS_TRAINING)
                    + "\titer: " + str(i) + "/" + str(max_iter)
                    + "\ttotal_iters: " + str(total_iters)
                    + "\td_loss:" + str(round(d_loss.item(), 4))
                    + "\tg_loss:" + str(round(g_loss.item(), 4))
                    + "\tc_loss:" + str(round(c_loss, 4))
                    )

        if (epoch + 1) % 5 == 0:
            generate_imgs(fixed_z, fixed_label, img_training_path, epoch=epoch + 1)

    torch.save(classifier.state_dict(), cls_trained)
    torch.save(gen.state_dict(), g_trained)
    torch.save(dis.state_dict(), d_trained)

    print('Finished Training GAN')
    print('\n')


# --------------------
#     Final Model
# --------------------

generate_imgs(fixed_z, fixed_label, img_training_path)

print(f'Target test accuracy on target: {100*accuracy(classifier, target_loader_test):.2f}%')
class_accuracy(classifier, target_loader_test, list(range(N_CLASSES)))