import os
import uuid
import argparse
import yaml
from torch import load, nn, optim, save
from torch.utils.tensorboard import SummaryWriter
from facetracker import FaceTracker
from utils import load_data, get_ciou, train_loop, test_loop, load_best_params, random_annotation

def train(
          model_name: str,
          learning_rate: float=1e-4,
          epochs: int=10,
          batch_size: int=64,
          optuna_study: str=None,
          data_root: str='data'
):
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
    optuna_study : str, optional
        Load the best hyperparameters from the specified Optuna study.
        Otherwise, uses default classification and bounding box loss weights (0.5, 0.5).
    data_root : str, optional
        Directory from which to read training data.

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
    if optuna_study is not None:
        try:
            learning_rate, cw, bw = load_best_params(optuna_study)
        except ValueError:
            print(f'The Optuna study {optuna_study} does not exist.')
            while True:
                cont = input('Should the model be trained on base hyperparameters? [Y/N] ').lower()
                if cont == 'y':
                    cw, bw = .5, .5
                    break
                elif cont == 'n':
                    print('Terminating training session...')
                    return
                else:
                    print('Please enter a valid character.\n')
    else:
        cw, bw = .5, .5

    model = FaceTracker()

    # Load existing model weights
    model_path = os.path.join('detection', 'model_weights', f'{model_name}.pth')
    if os.path.exists(model_path):
        print(f'Loading existing weights for {model_name}...\n')
        model.load_state_dict(load(model_path, weights_only=True))
    else:
        print(f'Saving new weights for {model_name}...\n')
        save(model.state_dict(), model_path)
        os.mkdir(os.path.join('detection', 'logs', 'facetracker', model_name))

    train_dataloader, test_dataloader, testing_data = load_data(batch_size, data_root)

    closs_fn = nn.BCELoss()
    lloss_fn = get_ciou

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=os.path.join('detection', 'logs', 'facetracker', model_name))

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop(
            dataloader=train_dataloader,
            model=model,
            closs_fn=closs_fn,
            lloss_fn=lloss_fn,
            optimizer=optimizer,
            epoch=t,
            batch_size=batch_size,
            model_name=model_name,
            writer=writer,
            class_weight=cw,
            bbox_weight=bw
        )
        
        class_accuracy, bbox_accuracy = test_loop(
            dataloader=test_dataloader,
            model=model,
            epoch=t,
            writer=writer
        )

        print("\nTest Error:")
        print(f"Class Accuracy: {(100.*class_accuracy):>0.1f}%, B-Box Accuracy: {(100.*bbox_accuracy):>0.1f}%\n")

        for _ in range(2):
            random_annotation(
                model,
                testing_data,
                os.path.join('detection', 'val_annotations', f'{uuid.uuid4()}.png')
            )

    print("Done!")

    writer.close()

def main(args):
    train(
        model_name=args.model_name,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optuna_study=args.optuna_study,
        data_root=args.data_root
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--optuna_study", type=str, default=None, help="Optuna study")
    parser.add_argument("--data_root", type=str, default="data", help="Top-level data directory")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(open(args.config))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    main(args)