from torch import load, nn, optim
from torch.utils.tensorboard import SummaryWriter
from models import FaceTracker
from utils import load_data, get_ciou, train_loop, test_loop

# Load existing model weights
model = FaceTracker()
model_name = 'firecracker'
model.load_state_dict(load(f'model_weights\\{model_name}.pth', weights_only=True))

# Specify training parameters
learning_rate = 1e-4
epochs = 10
batch_size = 64

train_dataloader, test_dataloader = load_data(batch_size)

# Establish loss functions
closs_fn = nn.BCELoss()
lloss_fn = get_ciou

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir='logs')

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