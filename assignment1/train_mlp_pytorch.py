################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_training_progress(logging_dict, title='Pytorch MLP'):
    """
    Creates two separate plots showing training loss and validation accuracy over epochs.

    Args:
        logging_dict: Dictionary containing 'train_epoch_loss' and 'val_accuracy' lists
        title: Title of the plots
    """
    epochs = range(1, len(logging_dict['train_epoch_loss']) + 1)

    # Plot training loss
    plt.figure()
    plt.plot(epochs, logging_dict['train_epoch_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title(f'{title}: Loss Curve')
    plt.legend()
    fig_name = title.lower().replace(' ', '_').replace('+', '_')
    plt.savefig(f'results/{fig_name}_loss.png')
    plt.show()

    # Plot validation accuracy
    plt.figure()
    plt.plot(epochs, logging_dict['val_accuracy'], 'orange', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{title}: Accuracy')
    plt.legend()

    # Save plot with title
    plt.savefig(f'results/{fig_name}_accuracy.png')
    plt.show()

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      targets: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch

    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Convert predictions and targets to numpy
    predictions = predictions.detach().numpy()
    targets = targets.detach().numpy()

    # Compute predicted classes
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_correct = 0
    total_samples = 0

    for batch_data, batch_targets in data_loader:
        # Flatten the whole batch
        batch_data = batch_data.view(batch_data.size(0), -1)
        # Forward pass
        predictions = model(batch_data)

        batch_accuracy = accuracy(predictions, batch_targets)
        total_correct += batch_accuracy * batch_data.shape[0]
        total_samples += batch_data.shape[0]

    # Compute average accuracy
    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_inputs = 32 * 32 * 3
    n_classes = 10

    # TODO: Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes, use_batch_norm).to(device)
    print(f"PyTorch Model: {model}")
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    val_accuracies = []
    # TODO: Test best model
    test_accuracy = 0.0
    # TODO: Add any information you might want to save for plotting
    logging_dict = {'train_loss': [], 'val_accuracy': [], 'train_epoch_loss': []}

    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_model = None
    best_val_accuracy = 0.0

    start_time = time.time()

    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        for inputs, targets in cifar10_loader['train']:
            # Flatten data
            inputs = inputs.reshape(inputs.shape[0], -1)
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = loss_module(outputs, targets)
            logging_dict['train_loss'].append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate model on validation set
        model.eval()
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)
        logging_dict['val_accuracy'].append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

        # Mean epoch loss
        mean_train_loss = np.mean(logging_dict['train_loss'][-len(cifar10_loader['train']):])
        logging_dict['train_epoch_loss'].append(mean_train_loss)

        # Print epoch loss
        print(f'Epoch {epoch + 1} - Training Loss: {mean_train_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

    # Print total training time
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')

    # Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'Test accuracy of best model: {test_accuracy:.4f}')


    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    # Custom title for the plot
    parser.add_argument('--plot_title', default='Pytorch MLP', type=str,
                        help='Custom title for the training progress plot')

    args = parser.parse_args()
    kwargs = vars(args)
    plot_title = kwargs.pop('plot_title')

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    # Plot training progress
    plot_training_progress(logging_dict, title=plot_title)

