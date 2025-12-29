import os
import uuid
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from facetracker import FaceTracker
from utils import utils

def train(
          model_name: str,
          learning_rate: float=1e-4,
          epochs: int=10,
          batch_size: int=64,
          optuna_study: str=None,
          data_root: str='data',
          parallelize: bool=False,
          num_annotations: int=0
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
            learning_rate, cw, bw = utils.load_best_params(optuna_study)
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
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f'Saving new weights for {model_name}...\n')
        torch.save(model.state_dict(), model_path)
        os.mkdir(os.path.join('detection', 'logs', 'facetracker', model_name))

    if parallelize:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        dist.init_process_group(backend=backend, init_method="env://")
        model.to(device)
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        rank = 0

    if rank == 0:
        log_dir = os.path.join('detection', 'logs', 'facetracker', model_name)
        writer = SummaryWriter(log_dir=log_dir)
        class_accuracy_history = []
        bbox_accuracy_history = []
    else:
        writer = None

    train_dataloader, test_dataloader, testing_data = utils.load_data(
        batch_size=batch_size,
        data_root=data_root,
        parallelize=parallelize,
        return_test_data=True
    )

    closs_fn = torch.nn.BCELoss()
    lloss_fn = utils.get_ciou

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        if parallelize and t > 0:
            train_dataloader, test_dataloader = utils.load_data(
                batch_size=batch_size,
                data_root=data_root,
                parallelize=parallelize,
                epoch=t
            )

        print(f"Epoch {t+1}\n-------------------------------")

        utils.train_loop(
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
            bbox_weight=bw,
            device=device,
            rank=rank
        )
        
        class_accuracy, bbox_accuracy = utils.test_loop(
            dataloader=test_dataloader,
            model=model,
            epoch=t,
            writer=writer,
            device=device
        )

        print("\nTest Error:")
        print(f"Class Accuracy: {(100.*class_accuracy):>0.1f}%, B-Box Accuracy: {(100.*bbox_accuracy):>0.1f}%\n")

        if rank == 0:
            class_accuracy_history.append(100. * class_accuracy)
            bbox_accuracy_history.append(100. * bbox_accuracy)

            if num_annotations > 0:
                for _ in range(num_annotations):
                    utils.random_annotation(
                        model=model,
                        dataset=testing_data,
                        img_dest=os.path.join('detection', 'val_annotations', f'{uuid.uuid4()}.png'),
                        device=device
                    )
        
        if parallelize:
            dist.barrier()

    print("Done!")

    if writer is not None:
        writer.close()
    
    if parallelize:
        dist.destroy_process_group()
    
    if rank == 0:
        utils.plot_accuracies(
            class_accuracy_history=class_accuracy_history,
            bbox_accuracy_history=bbox_accuracy_history,
            filepath=os.path.join(log_dir, f'{uuid.uuid4()}.png')
        )

def main(args):
    train(
        model_name=args.model_name,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optuna_study=args.optuna_study,
        data_root=args.data_root,
        parallelize=args.parallelize,
        num_annotations=args.num_annotations
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--optuna_study", type=str, default=None, help="Optuna study")
    parser.add_argument("--data_root", type=str, default="data", help="Top-level data directory")
    parser.add_argument("--parallelize", action="store_true", default=False, help="Parallelize training")
    parser.add_argument("--num_annotations", type=int, default=0, help="Number of test annotations")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(open(args.config))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    main(args)