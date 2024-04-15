import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def accuracy(model, dataloader, verbose=False):
    correct = 0
    total = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if verbose:
        print('Accuracy of the network: %d %%' % (100 * correct / total))
    else:
        return correct/total


def class_accuracy(model, dataloader, classes, verbose=False):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    device = next(model.parameters()).device

    # again no gradients needed
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class {classname} is: {accuracy:.1f}% ({total_pred[classname]} elements)")
    print('\n')
    

def cls_pretraining(classifier, loader_train, loader_test, learning_rate, n_epochs, results_path):
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
    optimizer.zero_grad()
    outputs = classifieur(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()