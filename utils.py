import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.custom_dataset import CustomDataset
import matplotlib.pyplot as plt

def get_data_loaders(train_dir, test_dir, train_label, test_label, batch_size, img_size):
    """
    Input parameters:
        train_dir: str, path to training images
        test_dir: str, path to testing images
        train_label: str, path to training labels
        test_label: str, path to testing labels
        batch_size: int, batch size
        img_size: tuple, image size
    Output:
        Return training and testing data loaders
    """
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CustomDataset(train_dir, train_label, transform=transform_train)
    test_dataset = CustomDataset(test_dir, test_label, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def plot_loss_curve(train_losses, val_losses, save_path):
    """
    Input parameters:
        train_losses: list, training losses
        val_losses: list, validation losses
        save_path: str, path to save the plot
    Output:
        Return the plot of training and validation losses
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()