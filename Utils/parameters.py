from hmac import new
import torch

class HyperParameters:  # ⚙️ Class for fine-tuning training nuances ⚙️
    """
    Manages and streamlines hyperparameters for a seamless training experience. ✨
    """

    def __init__(self):
        """
        Initializes hyperparameters with their default values. ️
        """

        #  Paths and directories 
        self.result_dir = "./user_data/"  #  Directory for storing precious results 
        self.data_dir = "../input/"  #  Directory housing essential input data 

        # ➗ Cross-validation settings ➗
        self.k_folds = 5  # ⚖️ Number of folds for rigorous cross-validation ⚖️
        self.n_jobs = 4  #  Number of parallel jobs for efficient cross-validation 

        #  Randomization and reproducibility 
        self.random_seed = 42  # ✨ Seed for ensuring consistent results across multiple runs ✨

        # ✂️ Data processing parameters ✂️
        self.seq_length = 2048  #  Maximum sequence length for optimal data shaping 
        self.cell_count = 128  #  Number of cells within a sequence 
        self.cell_max_length = 128  #  Maximum length allowed for each cell 

        #  Device configuration 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  #  Wisely selecting the most powerful device (GPU or CPU) 

        #  Model hyperparameters 
        self.use_cuda = torch.cuda.is_available()  # ⚡ Flag indicating GPU readiness ⚡
        self.gpu = 0  #  Index of the favored GPU (if multiple options exist) 

        #  Training hyperparameters 
        self.print_freq = 100  #  Frequency of progress updates during the training journey 
        self.lr = 0.003  # ‍♀️ Learning rate for model's adaptive strides ‍♀️
        self.weight_decay = 0  # ⚖️ Regularization technique to prevent overfitting ⚖️
        self.optim = "Adam"  #  The chosen optimizer, reigning supreme over model updates 
        self.base_epoch = 30  # ️ Foundational number of epochs for model training ️

    def get(self, name):
        """
        Retrieves a hyperparameter by name.

        Args:
            name (str): Name of the hyperparameter.

        Returns:
            Any: The value of the hyperparameter.
        """

        return getattr(self, name) # 🪄 Magically accesses the desired hyperparameter 🪄

    def set(self, **kwargs):
        """
        Sets multiple hyperparameters. ⚗️

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
   # Summon the hyperparameter wizard!
    parameters = HyperParameters()

    # Fetch a hyperparameter incantation
    lr = parameters.get("lr")
    print(f"Current learning rate: {lr}", end="\n\n")

    # Tweak your spells!
    parameters.set(learning_rate=0.005)

    # Behold the secrets of your hyperparameter kingdom!
    print("Updated hyperparameters:", parameters, sep="\n\n")