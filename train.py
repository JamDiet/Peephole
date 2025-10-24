from torch import load, nn, optim
from torch.utils.data import DataLoader
from models import FaceTracker
from utils import CustomImageDataset, ciou, train_loop

learning_rate = 1e-4
batch_size = 64
epochs = 10
model_name = 'firecracker'

# Load training data
training_data = CustomImageDataset('train')
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# Load existing model weights
model = FaceTracker()
model.load_state_dict(load(f'model_weights\\{model_name}.pth', weights_only=True))

# Establish loss functions
closs_fn = nn.BCELoss()
lloss_fn = ciou

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, closs_fn, lloss_fn, optimizer, batch_size, model_name)

print("Done!")