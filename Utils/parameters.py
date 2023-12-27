import torch

class HyperParameters:
   """
   Encapsulates and manages hyperparameters for a training process.
   """

   def __init__(self):
       """
       Initializes hyperparameters with default values.
       """

       # Paths and directories
       self.result_dir = "./user_data/"  # Directory for saving results
       self.data_dir = "../input/"  # Directory containing input data

       # Cross-validation settings
       self.k_folds = 5  # Number of folds for cross-validation
       self.n_jobs = 4  # Number of parallel jobs for cross-validation

       # Randomization and reproducibility
       self.random_seed = 42  # Random seed for reproducibility

       # Data processing parameters
       self.seq_length = 2048  # Maximum sequence length
       self.cell_count = 128  # Number of cells in a sequence
       self.cell_max_length = 128  # Maximum length of a cell

       # Device configuration
       self.device = torch.device(
           "cuda" if torch.cuda.is_available() else "cpu"
       )  # Device to use (GPU or CPU)

       # Model hyperparameters
       self.use_cuda = torch.cuda.is_available()  # Flag for GPU usage
       self.gpu = 0  # GPU index to use (if multiple GPUs available)

       # Training hyperparameters
       self.print_freq = 100  # Print frequency during training
       self.lr = 0.003  # Learning rate
       self.weight_decay = 0  # Weight decay
       self.optim = "Adam"  # Optimizer
       self.base_epoch = 30  # Base number of epochs

   def get(self, name):
       """
       Retrieves a hyperparameter by name.

       Args:
           name (str): Name of the hyperparameter.

       Returns:
           Any: The value of the hyperparameter.
       """

       return getattr(self, name)

   def set(self, **kwargs):
       """
       Sets multiple hyperparameters.

       Args:
           **kwargs: Keyword arguments where keys are hyperparameter names and values are their new values.
       """

       for k, v in kwargs.items():
           setattr(self, k, v)

   def __str__(self):
       """
       Returns a string representation of the hyperparameters.

       Returns:
           str: A formatted string containing hyperparameter names and values.
       """

       return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])


# Example usage
if __name__ == "__main__":
   parameters = HyperParameters()
   print(parameters)