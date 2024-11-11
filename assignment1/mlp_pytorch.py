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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.

        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Initialize the parent class
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm

        # Initialize the layers
        self.layers = OrderedDict()

        # Add the hidden layers
        if len(n_hidden) > 0:
            for i, hidden_units in enumerate(n_hidden):
                # Linear layer with Kaiming initialization
                self.layers[f'Hidden_{i}'] = nn.Linear(n_inputs, hidden_units)
                nn.init.kaiming_normal_(self.layers[f'Hidden_{i}'].weight)

                # Batch normalization
                if use_batch_norm:
                    self.layers[f'BatchNorm_{i}'] = nn.BatchNorm1d(hidden_units)

                # Activation function
                self.layers[f'ELU_{i}'] = nn.ELU()

                # Update the number of inputs
                n_inputs = hidden_units

        # Add the output layer (no activation function)
        self.layers['Output'] = nn.Linear(n_inputs, n_classes)
        nn.init.kaiming_normal_(self.layers['Output'].weight)

        # Instantiate the model
        self.model = nn.Sequential(self.layers)
        #######################
        # END OF YOUR CODE    #
        #######################

    def __str__(self):
        """
        Prints the architecture of the network.
        """
        return str(self.model)


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = self.model(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
