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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

import matplotlib.pyplot as plt
import numpy as np


def plot_training_progress(logging_dict):
    """
    Creates two plots: one showing training and validation loss over epochs,
    and another showing training and validation accuracy over epochs.

    Args:
        logging_dict: Dictionary containing 'train_losses', 'val_losses', 'train_accuracies', and 'val_accuracies' lists
    """
    # Plot training and validation loss
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(logging_dict['train_losses'], label='Training Loss', color='tab:blue')
    ax1.plot(logging_dict['val_losses'], label='Validation Loss', color='tab:orange')
    ax1.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('loss_progress.png')
    plt.close()

    # Plot training and validation accuracy
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.plot(logging_dict['train_accuracies'], label='Training Accuracy', color='tab:blue')
    ax2.plot(logging_dict['val_accuracies'], label='Validation Accuracy', color='tab:orange')
    ax2.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_progress.png')
    plt.close()


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      targets: 1D int array of size [batch_size], ground truth labels for each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e., the average correct predictions over the whole batch
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
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
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_correct = 0
    total_samples = 0

    for batch_data, batch_targets in data_loader:
        # Flatten data
        batch_data = batch_data.reshape(batch_data.shape[0], -1)

        # Forward pass
        predictions = model.forward(batch_data)

        batch_accuracy = accuracy(predictions, batch_targets)
        total_correct += batch_accuracy * batch_data.shape[0]
        total_samples += batch_data.shape[0]

    # Compute average accuracy
    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def calculate_loss_and_accuracy(model, data_loader, loss_module):
    """
    Calculates the loss and accuracy of the model on a given dataset.
    Args:
        model:
        data_loader:
        loss_module:

    Returns:

    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_data, batch_targets in data_loader:
        # Flatten data
        batch_data = batch_data.reshape(batch_data.shape[0], -1)

        # Forward pass
        predictions = model.forward(batch_data)

        # Calculate loss
        loss = loss_module.forward(predictions, batch_targets)
        total_loss += loss

        # Calculate accuracy
        batch_accuracy = accuracy(predictions, batch_targets)
        total_correct += batch_accuracy * batch_data.shape[0]
        total_samples += batch_data.shape[0]

    # Compute average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specifying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    # Initialize model and loss module
    n_inputs = 3 * 32 * 32  # CIFAR10 images: 3 channels, 32x32 pixels
    n_classes = 10
    model = MLP(n_inputs, hidden_dims, n_classes)
    # Print model
    print(model)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies = []
    # TODO: Test best model
    test_accuracy = 0.0
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'train_losses': [],
        'train_epoch_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'test_accuracy': 0.0,
        'best_validation_accuracy': 0.0
    }

    # Initialize best model
    best_model = None
    best_validation_accuracy = 0.0

    for epoch in tqdm(range(epochs), desc='Training'):
        # Training Phase
        total_train_loss, total_train_accuracy = 0.0, 0.0
        for batch_data, batch_targets in cifar10_loader['train']:
            batch_data = batch_data.reshape(batch_data.shape[0], -1)

            predictions = model.forward(batch_data)
            loss = loss_module.forward(predictions, batch_targets)
            logging_dict['train_epoch_losses'].append(loss)

            dL_dy = loss_module.backward(predictions, batch_targets)
            model.backward(dL_dy)

            for layer in model.layers:
                if hasattr(layer, 'params'):
                    for param_name, param in layer.params.items():
                        param -= lr * layer.grads[param_name]

        # Calculate loss and accuracy for the entire training set
        train_loss, train_accuracy = calculate_loss_and_accuracy(model, cifar10_loader['train'], loss_module)
        logging_dict['train_epoch_losses'].append(train_loss)
        logging_dict['train_accuracies'].append(train_accuracy)

        # Validation Phase
        val_loss, val_accuracy = calculate_loss_and_accuracy(model, cifar10_loader['validation'], loss_module)
        logging_dict['val_losses'].append(val_loss)
        logging_dict['val_accuracies'].append(val_accuracy)

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_model = deepcopy(model)

        logger.info(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Test Phase with best model
    test_loss, test_accuracy = calculate_loss_and_accuracy(best_model, cifar10_loader['test'], loss_module)
    logging_dict['test_accuracy'] = test_accuracy
    logging_dict['best_validation_accuracy'] = best_validation_accuracy
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

    args = parser.parse_args()
    kwargs = vars(args)

    # train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    # Train the model and get the logging dict
    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)

    # Create the plot
    plot_training_progress(logging_dict)

    # Print final results
    logger.info(f"Best validation accuracy: {logging_dict['best_validation_accuracy']:.4f}")
    logger.info(f"Test accuracy: {logging_dict['test_accuracy']:.4f}")
