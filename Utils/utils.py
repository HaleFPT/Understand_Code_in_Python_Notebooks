from fileinput import filename
import random
import pandas as pd
import numpy as np
import sys
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class KFold(object):
    """
    Implements K-fold cross-validation for data splitting.
    """

    def __init__(self, randomseed, k_fold=10, flag = 'fold_flag'):
        """
        Initializes the KFold object.

        Args:
            randomseed (int): Seed for random number generator.
            k_fold (int, optional): Number of folds. Defaults to 10.
            flag (str, optional): Name of the flag column to indicate fold assignments. Defaults to 'fold_flag'.
        """

        np.random.seed(randomseed) # Set random seed for reproducibility
        self.k_fold = k_fold # Store the number of folds
        self.flag_name = flag # Store the name of the flag column
    
    def group_split(self, train_df, group_cols):
        """
        Splits data into folds, keeping groups together within folds.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            group_cols (str or list): Columns to group by.

        Returns:
            pd.DataFrame: DataFrame with an additional column indicating fold assignments.
        """

        group_value = list(train_df[group_cols].values)
        group_value.sort()
        fold_flag = [i % self.k_fold for i in range(len(group_value))]
        np.random.shuffle(fold_flag)
        train_df = train_df.merge(pd.DataFrame({group_cols: group_value, self.flag_name: fold_flag}),
                                  how='left',
                                  on=group_cols)
        return train_df
    
    def random_split(self, train_df):
        """
        Splits data randomly into folds.

        Args:
            train_df (pd.DataFrame): Training DataFrame.

        Returns:
            pd.DataFrame: DataFrame with an additional column indicating fold assignments.
        """

        fold_flag = [i % self.k_fold for i in range(len(train_df))]
        np.random.shuffle(fold_flag)
        train_df[self.flag_name] = fold_flag
        return train_df
    
    def stratified_split(self, train_df, group_col):
        """
        Splits data into folds, maintaining class proportions within folds (stratification).

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            group_col (str): Column to stratify by.

        Returns:
            pd.DataFrame: DataFrame with an additional column indicating fold assignments.
        """

        train_df[self.flag_name] = 1
        train_df[self.flag_name] = train_df.groupby(by=[group_col])[self.flag_name].rank(ascending=True,
                                                                                         method="first").astype(int)
        train_df[self.flag_name] = train_df[self.flag_name].sample(frac=1).reset_index(drop=True)
        train_df[self.flag_name] = (train_df[self.flag_name]) % self.k_fold
        return train_df


class Logger(object):
    """
    A class for logging messages to both the terminal and a file.
    """

    def __init__(self):
        """
        Initializes the logger with default output to the terminal.
        """
        self.terminal = sys.stdout  # Redirect output to the terminal
        self.send_file = None  # File handle for logging to a file

    def open(self, file, mode=None):
        """
        Opens a file for logging.

        Args:
            file (str): The path to the file to open.
            mode (str, optional): The mode in which to open the file. Defaults to 'w' (write).
        """
        if mode is None:
            mode = 'w'  # Set default mode to write
        self.send_file = open(file, mode)  # Open the file with the specified mode

    def write(self, message, is_terminal=1, is_file=1):
        """
        Writes a message to the terminal and/or a file.

        Args:
            message (str): The message to write.
            is_terminal (int, optional): Whether to write to the terminal. Defaults to 1 (True).
            is_file (int, optional): Whether to write to the file. Defaults to 1 (True).
        """
        if '\r' in message:
            is_file = 0  # Prevent overwriting lines in the file if the message contains a carriage return
        if is_terminal == 1:
            self.terminal.write(message)  # Write to the terminal
            self.terminal.flush()  # Ensure the message is immediately displayed
            # time.sleep(1)  # Uncomment this line to add a 1-second delay between messages
        if is_file == 1:
            self.send_file.write(message)  # Write to the file
            self.send_file.flush()  # Ensure the message is written to disk

    def flush(self):
        """
        Flushes any pending output to the terminal and/or file.

        Note: This method is currently empty for Python 3 compatibility. You can add custom behavior if needed.
        """
        pass

    def seed_everything(random_seed):
        """
        Sets random seeds for reproducibility.

        Args:
            random_seed (int): The seed value to use.
        """
        random.seed(random_seed)  # Set random seed for Python's random library
        torch.manual_seed(random_seed)  # Set random seed for PyTorch
        os.environ['PYTHONHASHSEED'] = str(random_seed)  # Set Python hash seed
        if torch.cuda.is_available():  # If CUDA is available
            torch.cuda.manual_seed(random_seed)  # Set random seed for CUDA tensors
            torch.cuda.manual_seed_all(random_seed)  # Set random seed for all CUDA devices
            torch.backends.cudnn.deterministic = True  # Enable deterministic mode for CuDNN
            torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking

class AverageMeter(object):
   """
   Computes and stores the average and current value

   Example:
       >>> meter = AverageMeter()
       >>> meter.update(val=10, count=1)  # Adds a single value
       >>> meter.update(val=20, count=2)  # Adds a batch of 2 values
       >>> print(meter.val)       # Current value: 20
       >>> print(meter.avg)       # Average value: 15
   """

   def __init__(self):
       """
       Initializes the meter and resets all values.
       """
       self.reset()

   def reset(self):
       """
       Resets all variables to zero.
       """
       self.avg = 0.0  # Running average
       self.sum = 0.0  # Cumulative sum of values
       self.val = 0.0  # Most recently added value
       self.count = 0  # Total number of values added

   def update(self, val, count=1):
       """
       Updates the meter with a new value.

       Args:
           val (float): The new value to add.
           count (int, optional): The number of times to add the value. Defaults to 1.
       """
       self.val = val  # Set the most recent value
       self.sum += val * count  # Update the cumulative sum
       self.count += count  # Update the total count
       self.avg = self.sum / self.count  # Calculate the new average
    

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


class CustomLoss(nn.Module):
   """
   Custom loss function designed for a specific ranking task.

   Calculates the mean absolute error between predicted scores and ground truth ranks,
   considering only valid entries indicated by a rank mask.
   """

   def __init__(self):
       super(CustomLoss, self).__init__()

   def forward(self, inputs, rank, rank_mask):
       """
       Calculates the loss for a given batch of inputs, ranks, and rank masks.

       Args:
           inputs: Tensor of predicted scores, shape (batch_size, num_items).
           rank: Tensor of ground truth ranks, shape (batch_size, num_items).
           rank_mask: Boolean tensor indicating valid entries for loss calculation,
                        shape (batch_size, num_items).

       Returns:
           Tensor representing the mean absolute error loss.
       """

       # Calculate element-wise absolute errors, masking out invalid entries:
       masked_errors = torch.abs(inputs - rank) * rank_mask

       # Calculate the average error for each example in the batch:
       error_per_example = torch.sum(masked_errors, dim=1) / torch.sum(rank_mask, dim=1)

       # Return the mean error across all examples in the batch:
       return torch.mean(error_per_example)
   
class CustomBCELoss(nn.Module):
    """
    Custom binary cross-entropy loss function with optional class weighting and masking.

    Args:
        class_weights (bool, optional): Whether to apply class weights. Defaults to False.
    """

    def __init__(self, class_weights=False):
        super(CustomBCELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets, mask, sample_weights=None):
        """
        Calculates the loss.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_classes).
            targets (torch.Tensor): Target tensor of shape (batch_size, num_classes).
            mask (torch.Tensor): Mask tensor of shape (batch_size, num_classes).
            sample_weights (torch.Tensor, optional): Sample weights tensor of shape (batch_size). Defaults to None.

        Returns:
            torch.Tensor: The calculated loss.
        """

        # Print inputs for debugging (can be removed)
        print(inputs)

        # Select relevant inputs based on target shape
        inputs = inputs[:, targets.shape[1]]

        # Calculate binary cross-entropy for positive and negative cases separately
        bce1 = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
        bce2 = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

        # Combine BCEs based on targets
        bce = 1 * bce1 * targets + bce2 * (1 - targets)

        # Apply mask to exclude irrelevant elements
        mask = torch.where(targets >= 0, torch.ones_like(targets), torch.zeros_like(targets))
        bce = bce * mask

        # Print BCE for debugging (can be removed)
        print(bce)

        # Apply sample weights if provided
        if sample_weights is not None:
            bce = bce * sample_weights.unsqueeze(1)

        # Calculate mean loss across masked elements and then across the batch
        loss = torch.sum(bce, dim=1) / torch.sum(mask, dim=1)

        return torch.mean(loss)