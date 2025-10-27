import optuna
from torch import optim, nn
import numpy as np
from models import FaceTracker
from utils import load_data, train_loop, test_loop, get_ciou

class ObjectiveContainer():
    def __init__(self, train_dataloader, test_dataloader, closs_fn, lloss_fn, batch_size: int):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.closs_fn = closs_fn
        self.lloss_fn = lloss_fn
        self.batch_size = batch_size

    def objective(self, trial):
        # Hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
        class_weight = trial.suggest_float('class_weight', 0., 1.)
        bbox_weight = trial.suggest_float('bbox_weight', 0., 1.)

        model = FaceTracker(class_weight, bbox_weight)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loop(dataloader=self.train_dataloader,
                   model=model,
                   closs_fn=self.closs_fn,
                   lloss_fn=self.lloss_fn,
                   optimizer=optimizer,
                   batch_size=self.batch_size)
        
        class_accuracy, bbox_accuracy = test_loop(self.test_dataloader, model)

        return class_accuracy + bbox_accuracy - np.std([class_accuracy, bbox_accuracy])

if __name__ == '__main__':
    # Training parameters
    batch_size = 64
    closs_fn = nn.BCELoss()
    lloss_fn = get_ciou

    train_dataloader, test_dataloader = load_data(batch_size)

    obj = ObjectiveContainer(train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             closs_fn=closs_fn,
                             lloss_fn=lloss_fn,
                             batch_size=batch_size)
    
    study = optuna.create_study(study_name="facetracker_optimization_std",
                                storage="sqlite:///optuna_history.db",
                                direction="maximize",
                                load_if_exists=True)

    study.optimize(obj.objective, n_trials=100)