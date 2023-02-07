import numpy as np

class EarlyStopback:
    """Creates early stopback object.

    Args:
        patience (int, optional): Number of epochs to wait before stopping. Defaults to 2.
        min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Defaults to 0.05.

    Returns: nothing
      """

    def __init__(self, patience=2, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
