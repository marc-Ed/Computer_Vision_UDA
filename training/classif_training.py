"""
Pretraining functions for classifier and GAN
"""
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from util.metrics import accuracy


def cls_pretraining(classifier, loader_train, loader_test, learning_rate, n_epochs, results_path):
    """

    Args:
        classifier (TYPE):
        loader_train (TYPE):
        loader_test (TYPE):
        learning_rate (TYPE):
        n_epochs (TYPE):
        results_path (TYPE):

    Returns:
        Trained classifier on loader
    """

    # Pretraining paths
    cls_pretrained_path = ''.join([results_path, '/cls_pretrained.pth'])

    device = next(classifier.parameters()).device

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    os.makedirs(results_path, exist_ok=True)

    if os.path.isfile(cls_pretrained_path):
        classifier.load_state_dict(torch.load(cls_pretrained_path))
        print('loaded existing model')

    else:
        print('Starting Training classifier')
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for inputs, labels in loader_train:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = classifier_train_step(classifier, inputs, optimizer, criterion, labels)
                running_loss += loss

            print(f'Epoch: {epoch} || loss: {running_loss}')
            if (epoch + 1) % 10 == 0:
                print(f'Test accuracy: {100*accuracy(classifier, loader_test):.2f}%')
        print('Finished Training classifier')
        print('\n')

    print('Results:')
    print(f'Test accuracy on source: {100*accuracy(classifier, loader_test):.2f}%')
    torch.save(classifier.state_dict(), cls_pretrained_path)

    return classifier


def classifier_train_step(classifieur, inputs, optimizer, criterion, labels):
    """Summary

    Args:
        classifieur (TYPE)
        inputs (TYPE)
        optimizer (TYPE)
        criterion (TYPE)
        labels (TYPE)
        classifieur (TYPE)
        inputs (TYPE)
        optimizer (TYPE)
        criterion (TYPE)
        labels (TYPE)

    Returns:
        TYPE: loss value
    """
    optimizer.zero_grad()
    outputs = classifieur(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()