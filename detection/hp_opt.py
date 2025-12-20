import os
import argparse
import yaml
from optuna import create_study
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from numpy import std
from facetracker import FaceTracker
from detection.utils import utils

class ObjectiveContainer():
    def __init__(
                 self,
                 closs_fn,
                 lloss_fn,
                 batch_size: int=64,
                 num_epochs: int=10,
                 parallelize: bool=False,
                 data_root: str="data"
    ):
        self.closs_fn = closs_fn
        self.lloss_fn = lloss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.parallelize = parallelize
        self.data_root = data_root

        if parallelize:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.cuda.set_device(self.local_rank) if torch.cuda.is_available() else None
            dist.init_process_group(backend=backend, init_method="env://")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(self, trial):
        # Hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
        class_weight = trial.suggest_float('class_weight', 0., 1.)
        bbox_weight = trial.suggest_float('bbox_weight', 0., 1.)

        model = FaceTracker()
        model.to(self.device)

        if self.parallelize:
            model = DDP(model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_dataloader, test_dataloader = utils.load_data(
            batch_size=self.batch_size,
            data_root=self.data_root,
            parallelize=self.parallelize
        )

        for t in range(self.num_epochs):
            if self.parallelize and t > 0:
                train_dataloader, test_dataloader = utils.load_data(
                    batch_size=self.batch_size,
                    data_root=self.data_root,
                    epoch=t,
                    parallelize=self.parallelize
                )

            utils.train_loop(
                dataloader=train_dataloader,
                model=model,
                closs_fn=self.closs_fn,
                lloss_fn=self.lloss_fn,
                optimizer=optimizer,
                batch_size=self.batch_size,
                class_weight=class_weight,
                bbox_weight=bbox_weight,
                device=self.device
            )

            if self.parallelize:
                dist.barrier()
        
        class_accuracy, bbox_accuracy = utils.test_loop(
            dataloader=test_dataloader,
            model=model,
            device=self.device
        )

        print("\nTest Error:")
        print(f"Class Accuracy: {(100.*class_accuracy):>0.1f}%, B-Box Accuracy: {(100.*bbox_accuracy):>0.1f}%\n")

        return class_accuracy + bbox_accuracy - std([class_accuracy, bbox_accuracy])
    
    def destroy_process_group(self):
        if self.parallelize:
            dist.destroy_process_group()

class StudyContainer():
    def __init__(
            self,
            study_name: str,
            n_trials: int=100,
            num_epochs: int=10,
            data_root: str='data',
            batch_size: int=64,
            pool: bool=False,
            ddp: bool=False
    ):
        self.study_name = study_name
        self.n_trials = n_trials
        self.num_epochs = num_epochs
        self.data_root = data_root
        self.batch_size = batch_size

        # Only run 1 parallelization framework at a time
        if pool == True and ddp == True:
            pool = False
        self.pool = pool
        self.ddp = ddp

    def run_study(self, _):
        # Training parameters
        closs_fn = torch.nn.BCELoss()
        lloss_fn = utils.get_ciou

        obj = ObjectiveContainer(
            closs_fn=closs_fn,
            lloss_fn=lloss_fn,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            parallelize=self.ddp,
            data_root=self.data_root
        )
        
        if self.pool:
            study = create_study(
                study_name=self.study_name,
                storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
                direction="maximize",
                load_if_exists=True
            )
        else:
            study = create_study(
                study_name=self.study_name,
                storage="sqlite:///detection/logs/optuna.db",
                direction="maximize",
                load_if_exists=True
            )

        study.optimize(obj.objective, n_trials=self.n_trials)

        obj.destroy_process_group()

def main(args):
    study = StudyContainer(
        study_name=args.study_name,
        n_trials=args.n_trials,
        num_epochs=args.num_epochs,
        data_root=args.data_root,
        batch_size=args.batch_size,
        pool=args.pool,
        ddp=args.ddp
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.pool:
        with Pool(processes=world_size) as pool:
            pool.map(study.run_study, range(args.n_trials))
    else:
        study.run_study()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--study_name", type=str, help="Study name")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--data_root", type=str, default='data', help="Top-level data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--pool", action="store_true", default=False, help="Optuna pooling flag")
    parser.add_argument("--ddp", action="store_true", default=False, help="DistributedDataParallel flag")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(open(args.config))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    main(args)