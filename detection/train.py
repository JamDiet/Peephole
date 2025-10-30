import os
from torch import load, nn, optim
from torch.utils.tensorboard import SummaryWriter
from facetracker import FaceTracker
from utils import load_data, get_ciou, train_loop, test_loop, load_best_params

def train(model_name: str,
          learning_rate: float=1e-4,
          epochs: int=10,
          batch_size: int=64,
          optuna_hp: bool=False):
    """
    Train a FaceTracker model for object classification and bounding box prediction.

    Parameters
    ----------
    model_name : str
        The name of the model. Used to locate or save model weights (e.g., 'facetracker_v1').
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    epochs : int, optional
        Number of training epochs. Default is 10.
    batch_size : int, optional
        Number of samples per batch. Default is 64.
    optuna_hp : bool, optional
        If True, load the best hyperparameters from a saved Optuna study.
        Otherwise, uses default classification and bounding box loss weights (0.5, 0.5).

    Notes
    -----
    - The function loads existing model weights if available.
    - It initializes dataloaders for training and testing.
    - Uses binary cross-entropy loss for classification and CIOU for bounding box regression.
    - Logs metrics and loss values to TensorBoard.
    - Prints classification and bounding box accuracy after each epoch.

    Returns
    -------
    None
        This function trains the model and logs training/testing results.
    """
    # Get best hyperparameters from Optuna study
    if optuna_hp:
        learning_rate, cw, bw = load_best_params()
    else:
        cw, bw = .5, .5

    model = FaceTracker(cw, bw)

    # Load existing model weights
    model_path = f'detection\\model_weights\\{model_name}.pth'
    if os.path.exists(model_path):
        model.load_state_dict(load(model_path, weights_only=True))

    train_dataloader, test_dataloader = load_data(batch_size)

    closs_fn = nn.BCELoss()
    lloss_fn = get_ciou

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir='detection\\logs\\facetracker')

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop(dataloader=train_dataloader,
                model=model,
                closs_fn=closs_fn,
                lloss_fn=lloss_fn,
                optimizer=optimizer,
                epoch=t,
                batch_size=batch_size,
                model_name=model_name,
                writer=writer)
        
        class_accuracy, bbox_accuracy = test_loop(test_dataloader, model, writer)

        print("Test Error:")
        print(f"Class Accuracy: {(100.*class_accuracy):>0.1f}%, B-Box Accuracy: {(100.*bbox_accuracy):>0.1f}%\n")

    print("Done!")

    writer.close()

if __name__ == '__main__':
    train(model_name='firecracker',
          learning_rate=1e-4,
          epochs=10,
          batch_size=64,
          optuna_hp=False)