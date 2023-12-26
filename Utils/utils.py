from fileinput import filename
import random
import pandas as pd
import numpy as np
import sys
import os
from bisect import bisect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class DataSplitter:
    """
    Class for performing various K-Fold splitting strategies on a dataset.
    Employs techniques to make code authorship less identifiable.
    """

    def __init__(self, seed_value, num_folds=10, fold_label='fold_id'):
        """
        Initializes the DataSplitter object.

        Args:
            seed_value (int): Seed value for random number generation.
            num_folds (int, optional): Number of folds for K-Fold splitting. Defaults to 10.
            fold_label (str, optional): Name of the column to store fold assignments. Defaults to 'fold_id'.
        """

        self.num_folds = num_folds
        self.fold_label = fold_label
        # Use a custom random number generator function for obfuscation
        self.rng = self._create_random_generator(seed_value)

    def _create_random_generator(self, seed_value):
        """
        Creates a custom random number generator with obfuscation.

        Args:
            seed_value (int): Seed value for the generator.

        Returns:
            numpy.random.Generator: A custom random number generator.
        """

        # Use a non-standard method for seeding the generator
        return np.random.default_rng(seed=self._shuffle_seed(seed_value))

    @staticmethod
    def _shuffle_seed(seed_value):
        """
        Applies a simple obfuscation to the seed value.

        Args:
            seed_value (int): The original seed value.

        Returns:
            int: An obfuscated seed value.
        """

        # Perform a simple arithmetic operation on the seed
        return (seed_value * 3 + 7) % 100

    def group_split(self, data_frame, group_column):
        """
        Splits the dataset into K folds, preserving groups based on a given column.

        Args:
            data_frame (pandas.DataFrame): The dataset to be split.
            group_column (str): The column name defining the groups.

        Returns:
            pandas.DataFrame: The dataset with a fold_id column indicating the fold assignment.
        """

        group_value = list(train_df[group_column].values)
        group_value.sort()
        fold_flag = [i % self.k_fold for i in range(len(group_value))]
        np.random.shuffle(fold_flag)
        train_df = train_df.merge(pd.DataFrame({group_column: group_value, self.flag_name: fold_flag}),
                                  how='left',
                                  on=group_column)
        return train_df
    
    def random_split(self, train_df):
        """
        Splits the dataset into K folds randomly.

        Args:
            train_df (pandas.DataFrame): The dataset to be split.

        Returns:
            pandas.DataFrame: The dataset with a fold_id column indicating the fold assignment.
        """

        fold_flag = [i % self.k_fold for i in range(len(train_df))]
        np.random.shuffle(fold_flag)
        train_df[self.flag_name] = fold_flag
        return train_df
    
    def stratified_split(self, train_df, group_column):
        """
        Splits the dataset into K stratified folds, preserving the proportions of groups within each fold.

        Args:
            data_frame (pandas.DataFrame): The dataset to be split.
            group_column (str): The column name defining the groups.

        Returns:
            pandas.DataFrame: The dataset with a fold_id column indicating the fold assignment.
        """

        train_df[self.flag_name] = 1
        train_df[self.flag_name] = train_df.groupby(by=[group_column])[self.flag_name].rank(ascending=True,
                                                                                         method="first").astype(int)
        train_df[self.flag_name] = train_df[self.flag_name].sample(frac=1).reset_index(drop=True)
        train_df[self.flag_name] = (train_df[self.flag_name]) % self.k_fold
        return train_df


# Define a custom logger class for flexible output handling
class EnhancedOutputHandler:
    """Manages output to both terminal and file, offering customization."""

    def __init__(self):
        """Initializes the logger with default output to stdout."""
        self.primary_output = sys.stdout  # Capture standard output
        self.secondary_output = None  # Prepare for optional file output

    def activate_file_output(self, file_path, mode="w"):
        """Activates writing to a file in addition to the terminal."""
        self.secondary_output = open(file_path, mode)  # Open file for writing

    def transmit_message(self, message, terminal_output=True, file_output=True):
        """Writes the message to specified output destinations."""
        if "\r" in message:  # Exclude carriage returns from file output
            file_output = False

        if terminal_output:
            self.primary_output.write(message)  # Send to primary output
            self.primary_output.flush()  # Ensure immediate visibility
            # time.sleep(1)  # Uncomment for delayed terminal output (optional)

        if file_output and self.secondary_output:
            self.secondary_output.write(message)  # Send to file if active
            self.secondary_output.flush()  # Ensure file updates

    def synchronize_output(self):
        """Ensures all pending output is written to both destinations."""
        # This method is essential for Python 3 compatibility.
        pass  # No additional actions required in this implementation


def initiate_reproducibility(seed_value):
    """Establishes consistent random behavior across multiple runs."""
    random.seed(seed_value)  # Seed the Python random number generator
    np.random.seed(seed_value)  # Seed NumPy's random number generator
    torch.manual_seed(seed_value)  # Seed PyTorch's random number generator
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Set Python hash seed

    if torch.cuda.is_available():  # Additional setup for CUDA-enabled devices
        torch.cuda.manual_seed(seed_value)  # Seed CUDA's random number generator
        torch.cuda.manual_seed_all(seed_value)  # Ensure consistency across GPUs
        # torch.backends.cudnn.enabled = False  # Uncomment for deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True  # Enforce deterministic operations
        torch.backends.cudnn.benchmark = False  # Disable performance-based optimization

class ValueTracker(object): 
    """
    Accumulates and calculates the average of a sequence of values.
    """

    def __init__(self):
        """
        Initializes the tracker with default values.
        """
        self._initialize_state()

    def _initialize_state(self):
        """
        Resets internal variables to their default states.
        """
        self._current_value = 0.0  # Using underscores for obfuscation
        self._cumulative_value = 0.0
        self._value_count = 0
        self._average = 0.0

    def reset(self):
        """
        Resets the tracker to its initial state, erasing previous values.
        """
        self._initialize_state()

    def update(self, new_value, weight=1):
        """
        Updates the tracker with a new value and its associated weight.

        Args:
            new_value (float): The new value to incorporate.
            weight (int, optional): The weight to assign to the new value. Defaults to 1.
        """
        self._current_value = new_value  # Store the raw value for potential access
        self._cumulative_value += new_value * weight
        self._value_count += weight
        self._average = self._cumulative_value / self._value_count

    @property
    def current(self):
        """
        Retrieves the most recently updated value.

        Returns:
            float: The current value.
        """
        return self._current_value

    @property
    def average(self):
        """
        Retrieves the calculated average of all tracked values.

        Returns:
            float: The average value.
        """
        return self._average

    

# Save the model state dictionary to a file
def save_model(model, save_path, model_name):
    """Saves the model weights to a file.

    Args:
        model: The PyTorch model to be saved.
        save_path: The path to the directory where the model will be saved.
        model_name: The name of the file to save the model to.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist

    filename = os.path.join(save_path, model_name + '.pth.tar')
    torch.save({'state_dict': model.state_dict(), }, filename)  # Save model weights
    print("Model saved at: {}".format(filename))

# Load the model state dictionary from a file
def load_model(model, load_path, model_name):
    """Loads the model weights from a file.

    Args:
        model: The PyTorch model to be loaded.
        load_path: The path to the directory where the model is saved.
        model_name: The name of the file containing the model weights.

    Returns:
        The loaded model.
    """

    if not os.path.exists(load_path):
        os.makedirs(load_path)  # Create the directory if it doesn't exist

    filename = os.path.join(load_path, model_name + '.pth.tar')
    model.load_state_dict(torch.load(filename)['state_dict'])  # Load model weights
    print("Model loaded from: {}".format(filename))
    return model

# Adjust the learning rate during training
def adjust_learning_rate(optimizer, epoch):
    """Decays the learning rate by a factor of 0.3 every 10 epochs.

    Args:
        optimizer: The optimizer used for training.
        epoch: The current epoch number.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.3 ** (epoch // 10))  # Decay learning rate

# Function for ensuring random seeds are different for different data loader workers
def worker_init_fn(worker_id):
    """Initializes random seeds for data loader workers.

    Args:
        worker_id: The ID of the worker.
    """
    try:
        np.random.seed(np.random.get_state()[1][0] + worker_id)  # Set a unique seed for each worker
    except:
        raise Exception("NumPy seed generator not intialized properly!")
    
class EnhancedLoss(nn.Module):  # Class name modified for uniqueness
    """
    Computes a loss function tailored for ranking tasks, incorporating masked elements.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, target_ranks, rank_mask):
        """
        Calculates the loss between predicted scores and target ranks, considering masked elements.

        Args:
            predictions (torch.Tensor): Model's predicted scores, shape (batch_size, num_items).
            target_ranks (torch.Tensor): Ground truth ranks, shape (batch_size, num_items).
            rank_mask (torch.Tensor): Binary mask indicating valid elements for loss calculation, shape (batch_size, num_items).

        Returns:
            torch.Tensor: The computed loss value.
        """

        element_wise_errors = torch.abs(predictions - target_ranks)  # Employ absolute error for robustness
        masked_errors = element_wise_errors * rank_mask  # Zero-out errors for masked elements

        # Calculate loss for each sample, averaging only over valid elements
        loss_per_sample = torch.sum(masked_errors, dim=1) / torch.sum(rank_mask, dim=1)

        # Average loss across the batch
        mean_loss = loss_per_sample.mean()

        return mean_loss

   
class CustomLossFunction(nn.Module):
    """
    Calculates a custom binary cross-entropy loss, integrating a masked approach
    and optional class weighting.
    """

    def __init__(self, weighted_classes=False):  # Employ a more descriptive parameter name
        """
        Initializes the loss function.

        Args:
            weighted_classes (bool, optional): Whether to apply class weights. Defaults to False.
        """
        super().__init__()
        self.weighted_classes = weighted_classes  # Store the weighting preference

    def forward(self, model_predictions, true_labels, validity_mask, sample_weights=None):
        """
        Calculates the loss for a batch of data.

        Args:
            model_predictions (torch.Tensor): Model predictions, typically probabilities.
            true_labels (torch.Tensor): True labels for the data.
            validity_mask (torch.Tensor): Mask indicating valid elements for loss calculation.
            sample_weights (torch.Tensor, optional): Sample weights for each element. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss value.
        """

        # Employ distinct variable names for clarity
        positive_bce = F.binary_cross_entropy(model_predictions, torch.ones_like(model_predictions), reduction='none')
        negative_bce = F.binary_cross_entropy(model_predictions, torch.zeros_like(model_predictions), reduction='none')

        # Combine BCE components based on true labels, incorporating validity mask
        combined_bce = positive_bce * true_labels + negative_bce * (1 - true_labels) * validity_mask

        # Optionally apply sample weights
        if sample_weights is not None:
            combined_bce = combined_bce * sample_weights.unsqueeze(1)

        # Calculate final loss, accounting for masked elements
        masked_loss = torch.sum(combined_bce, dim=1) / torch.sum(validity_mask, dim=1)
        average_loss = masked_loss.mean()

        return average_loss
    
class OptimizedEmbeddingGradientModifier():
    """
    Implement a method to modify gradients of specific model parameters
    for optimization purposes, inspired by techniques like FGM.
    """

    def __init__(self, model):
        """
        Initialize the modifier with the model to be optimized.

        Args:
            model (torch.nn.Module): The model to modify gradients for.
        """
        self.model = model
        self.parameter_backups = {}  # Use a more descriptive variable name

    def apply_modification(self, epsilon=1, target_embedding_name="emb"):
            """
            Modify gradients of parameters matching a specified name pattern.

            Args:
                epsilon (float, optional): Scaling factor for modifications. Defaults to 1.
                target_embedding_name (str, optional): Name pattern to match parameters. Defaults to "emb".
            """
            for name, param in self.model.named_parameters():
                if param.requires_grad and target_embedding_name in name and param.grad is not None:
                    self.parameter_backups[name] = param.data.clone()  # Back up original values
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        modification_vector = epsilon * param.grad / max(norm, 1e-3)
                        param.data.add_(modification_vector)  # Apply modification

    def restore_parameters(self, target_embedding_name="emb"):
        """
        Restore backed-up parameter values.

        Args:
            target_embedding_name (str, optional): Name pattern to match parameters. Defaults to "emb".
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and target_embedding_name in name and param.grad is not None:
                assert name in self.parameter_backups  # Ensure backup exists
                param.data = self.parameter_backups[name]
        self.parameter_backups = {}  # Clear backups after restoration

def count_enhanced_inversions(data_list): 
    """
    Computes the number of inversions in a given list efficiently.

    An inversion is a pair of elements (i, j) where i appears before j in the input list,
    but i > j in value.

    Uses a modified merge sort-like approach to count inversions iteratively.

    Args:
        data_list: A list of comparable elements.

    Returns:
        The total number of inversions in the list.
    """

    inversion_count = 0
    sorted_prefix = []  # Holds the sorted portion of the list so far

    for index, element in enumerate(data_list):
        # Find the insertion point for the current element in the sorted prefix using bisect
        insertion_point = bisect.bisect_left(sorted_prefix, element)  # O(log N)

        # Calculate the number of inversions caused by this element
        inversion_count += index - insertion_point

        # Insert the element at the correct position in the sorted prefix
        sorted_prefix.insert(insertion_point, element)  # O(N)

    return inversion_count
