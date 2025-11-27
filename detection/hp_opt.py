from optuna import create_study
from torch import optim, nn
from numpy import std
from facetracker import FaceTracker
from utils import load_data, train_loop, test_loop, get_ciou

class ObjectiveContainer():
    def __init__(
                 self,
                 train_dataloader,
                 test_dataloader,
                 closs_fn,
                 lloss_fn,
                 batch_size: int,
                 num_epochs: int=1
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.closs_fn = closs_fn
        self.lloss_fn = lloss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def objective(self, trial):
        # Hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
        class_weight = trial.suggest_float('class_weight', 0., 1.)
        bbox_weight = trial.suggest_float('bbox_weight', 0., 1.)

        model = FaceTracker()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for _ in range(self.num_epochs):
            train_loop(
                dataloader=self.train_dataloader,
                model=model,
                closs_fn=self.closs_fn,
                lloss_fn=self.lloss_fn,
                optimizer=optimizer,
                batch_size=self.batch_size,
                class_weight=class_weight,
                bbox_weight=bbox_weight
            )
        
        class_accuracy, bbox_accuracy = test_loop(self.test_dataloader, model)

        print("\nTest Error:")
        print(f"Class Accuracy: {(100.*class_accuracy):>0.1f}%, B-Box Accuracy: {(100.*bbox_accuracy):>0.1f}%\n")

        return class_accuracy + bbox_accuracy - std([class_accuracy, bbox_accuracy])

def run_study(
        study_name: str,
        n_trials: int=100,
        num_epochs: int=10,
        data_root: str='data'
):
    # Training parameters
    batch_size = 64
    closs_fn = nn.BCELoss()
    lloss_fn = get_ciou

    train_dataloader, test_dataloader, _ = load_data(batch_size, data_root)

    obj = ObjectiveContainer(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        closs_fn=closs_fn,
        lloss_fn=lloss_fn,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    study = create_study(
        study_name=study_name,
        storage="sqlite:///detection/logs/optuna.db",
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(obj.objective, n_trials=n_trials)

if __name__ == '__main__':
    run_study(
        study_name='firecracker',
        n_trials=100,
        num_epochs=10,
        data_root='data'
    )