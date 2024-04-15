"""
Dataloader functions
"""
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

__all__ = ["load_mnist", "load_svhn"]

def interpolate_and_noise(img, noise_level=0.1):
    """ Apply a random interpolation between the original and inverted image, then add noise. """
    inverted_img = 1 - img
    alpha = random.uniform(0, 1)  # Random interpolation coefficient
    interpolated_img = img * (1 - alpha) + inverted_img * alpha
    
    # Add Gaussian noise
    noise = torch.randn_like(interpolated_img) * noise_level
    noisy_interpolated_img = interpolated_img + noise
    noisy_interpolated_img = torch.clamp(noisy_interpolated_img, 0, 1)  # Ensure pixel values are valid
    return noisy_interpolated_img

class CustomTransform:
    """ Custom transformation class for interpolating and adding noise to images. """
    def __init__(self, img_size, noise_level=0.1):
        self.img_size = img_size
        self.noise_level = noise_level

    def __call__(self, img):
        # Resize and normalize the image
        img = transforms.Resize((self.img_size, self.img_size))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
        return interpolate_and_noise(img, self.noise_level)

def load_mnist(data_path, img_size=28, batch_size=32, noise_level=0.1):
    """Load MNIST dataset with custom transformations for domain adaptation."""
    transform = CustomTransform(img_size, noise_level)
    
    mnist_dataset_train = MNIST(root=data_path, train=True, transform=transform, download=True)
    mnist_dataset_test = MNIST(root=data_path, train=False, transform=transform, download=True)
    
    mnist_loader_train = DataLoader(dataset=mnist_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    mnist_loader_test = DataLoader(dataset=mnist_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return mnist_loader_train, mnist_loader_test


def load_svhn(data_path, img_size=28, batch_size=32):
    """Summary

    Args:
        data_path (str)
        img_size (int, optional)
        batch_size (int, optional)

    Returns:
        svhn_loader_train (DataLoader): DataLoader for SVHN Train set
        svhn_loader_test (DataLoader): DataLoader for SVHN Test set
    """
    svhn_dataset_train = SVHN(root=data_path, split='train',
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.ToPILImage(),
                                                            transforms.Resize(
                                  (img_size, img_size)),
                                  transforms.Grayscale(
                                  num_output_channels=1),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                  mean=[0.5], std=[0.5])
                              ]),

                              download=True)

    svhn_dataset_test = SVHN(root=data_path, split='test',
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.ToPILImage(),
                                                           transforms.Resize(
                                                          (img_size, img_size)),
                                 transforms.Grayscale(
                                 num_output_channels=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                 mean=[0.5], std=[0.5])
                             ]),
                             download=True)

    svhn_loader_train = DataLoader(
        dataset=svhn_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    svhn_loader_test = DataLoader(
        dataset=svhn_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return svhn_loader_train, svhn_loader_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from data.datasets import data_sample

    DATA_PATH = '../datasets'
    mnist_train, mnist_test = load_mnist(DATA_PATH)
    svhn_train, svhn_test = load_svhn(DATA_PATH)

    data_sample(mnist_train)
    plt.suptitle('MNIST Train', fontsize=16)
    data_sample(mnist_test)
    plt.suptitle('MNIST Test', fontsize=16)
    data_sample(svhn_train)
    plt.suptitle('SVHN Train', fontsize=16)
    data_sample(svhn_test)
    plt.suptitle('SVHN Test', fontsize=16)
    plt.show()
