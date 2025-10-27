import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torch.utils.tensorboard import SummaryWriter

class CustomImageDataset(Dataset):
    '''Create PyTorch image dataset for single class and bounding box predictions.'''
    def __init__(self, partition: str):
        data_dir = os.path.join('data', partition)
        self.label_dir = os.path.join('data', partition, 'labels')
        self.img_labels = os.listdir(self.label_dir)
        self.img_dir = os.path.join(data_dir, 'images')
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        '''
        Returns
        -------
        image : torch.Tensor
            Tensor of shape (3, H, W) — normalized pixel values.
        class_label : torch.Tensor
            Tensor of shape (1,) — class probability.
        bbox_label : torch.Tensor
            Tensor of shape (4,) — bounding box coordinates (x_min, y_min, x_max, y_max).
        '''
        img_path = os.path.join(self.img_dir, self.img_labels[idx].split('.json')[0] + '.jpg')
        image = decode_image(img_path) / 255.

        with open(os.path.join(self.label_dir, self.img_labels[idx]), 'r', encoding = "utf-8") as f:
            label = json.load(f)
        class_label = torch.tensor(label['class'], dtype=torch.float32)
        bbox_label = torch.tensor(label['bbox'], dtype=torch.float32)
        
        return image, class_label, bbox_label
    
def load_data(batch_size: int):
    '''
    Load training and testing data.

    Parameters
    ----------
    batch_size : int
        Batch size for output DataLoaders.

    Returns
    -------
    train_dataloader : DataLoader
        PyTorch DataLoader containing training images.
    test_dataloader : DataLoader
        PyTorch DataLoader containing testing images.
    '''
    training_data = CustomImageDataset('train')
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    testing_data = CustomImageDataset('test')
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_iou(y_true, yhat):
    """
    Intersection over Union (IoU) for bounding box prediction accuracy.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth bounding box tensor of shape (4,).
    yhat : torch.Tensor
        Predicted bounding box tensor of shape (4,).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the IoU.
    """
    eps = 1e-7  # Prevent division by zero

    # Get intersection bounding box coordinates
    x_left = torch.maximum(y_true[:, 0], yhat[:, 0])
    y_top = torch.maximum(y_true[:, 1], yhat[:, 1])
    x_right = torch.minimum(y_true[:, 2], yhat[:, 2])
    y_bottom = torch.minimum(y_true[:, 3], yhat[:, 3])
    
    # Calculate intersection bounding box dimensions
    i_width = torch.clamp(x_right - x_left, min=0.)
    i_height = torch.clamp_(y_bottom - y_top, min=0.)
    i_area = i_width * i_height

    # Calculate ground truth dimensions
    true_width = y_true[:, 2] - y_true[:, 0]
    true_height = y_true[:, 3] - y_true[:, 1]
    true_area = true_width * true_height

    # Calculate prediction dimensions
    pred_width = yhat[:, 2] - yhat[:, 0]
    pred_height = yhat[:, 3] - yhat[:, 1]
    pred_area = pred_width * pred_height

    union = true_area + pred_area - i_area

    return i_area / (union + eps)

    
def get_ciou(y_true, yhat):
    """
    Complete Intersection over Union (CIoU) loss for bounding box prediction.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth bounding box tensor of shape (4,).
    yhat : torch.Tensor
        Predicted bounding box tensor of shape (4,).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the CIoU loss.
    """
    eps = 1e-7  # Prevent division by zero

    # ======= Compute IoU =======

    # Get intersection bounding box coordinates
    x_left = torch.maximum(y_true[:, 0], yhat[:, 0])
    y_top = torch.maximum(y_true[:, 1], yhat[:, 1])
    x_right = torch.minimum(y_true[:, 2], yhat[:, 2])
    y_bottom = torch.minimum(y_true[:, 3], yhat[:, 3])
    
    # Calculate intersection bounding box dimensions
    i_width = torch.clamp(x_right - x_left, min=0.)
    i_height = torch.clamp_(y_bottom - y_top, min=0.)
    i_area = i_width * i_height

    # Calculate ground truth dimensions
    true_width = y_true[:, 2] - y_true[:, 0]
    true_height = y_true[:, 3] - y_true[:, 1]
    true_area = true_width * true_height

    # Calculate prediction dimensions
    pred_width = yhat[:, 2] - yhat[:, 0]
    pred_height = yhat[:, 3] - yhat[:, 1]
    pred_area = pred_width * pred_height

    union = true_area + pred_area - i_area

    iou = i_area / (union + eps)

    # ======= Compute distance loss =======

    # Get coordinates of bounding box centers
    center1_x = 0.5 * (y_true[:, 2] + y_true[:, 0])
    center1_y = 0.5 * (y_true[:, 3] + y_true[:, 1])
    center2_x = 0.5 * (yhat[:, 2] + yhat[:, 0])
    center2_y = 0.5 * (yhat[:, 3] + yhat[:, 1])

    # Squared distance between centers
    d_sqr = torch.square(center2_x - center1_x) + torch.square(center2_y - center1_y)

    # Get corners of rectangle enveloping bounding boxes
    c_left = torch.minimum(y_true[:, 0], yhat[:, 0])
    c_top = torch.minimum(y_true[:, 1], yhat[:, 1])
    c_right = torch.maximum(y_true[:, 2], yhat[:, 2])
    c_bottom = torch.maximum(y_true[:, 3], yhat[:, 3])
    
    # Squared distance between corners
    c_sqr = torch.square(c_right - c_left) + torch.square(c_bottom - c_top)

    d_loss = d_sqr / (c_sqr + eps)

    # ======= Compute aspect ratio loss ========

    v = (4 / (math.pi ** 2)) * torch.square(
        torch.atan(true_width / (true_height + eps)) - 
        torch.atan(pred_width / (pred_height + eps))
    )

    alpha = v / (1 - iou + v + eps)
    
    ar_loss = alpha * v

    return torch.mean(1 - iou + d_loss + ar_loss)
    
def train_loop(dataloader,
               model,
               closs_fn,
               lloss_fn,
               optimizer,
               batch_size: int,
               epoch: int=None,
               model_name: str=None,
               writer: SummaryWriter=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    for batch, (X, y, z) in enumerate(dataloader):
        # Get predictions
        class_pred, bbox_pred = model(X)
        class_pred = class_pred.squeeze(1)

        # Calculate class and localization losses
        closs = closs_fn(class_pred, y)
        lloss = lloss_fn(z, bbox_pred)
        loss = closs + lloss

        # Log losses
        if writer != None and epoch != None:
            step = epoch * num_batches + batch
            writer.add_scalar('closs', closs, step)
            writer.add_scalar('lloss', lloss, step)
            writer.add_scalar('loss', loss, step)

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            # Display losses
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"closs: {closs:>7f}  lloss: {lloss:>7f}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Save weights
            if model_name != None:
                torch.save(model.state_dict(), f'model_weights\\{model_name}.pth')

def test_loop(dataloader, model, epoch: int=None, writer: SummaryWriter=None):
    model.eval()
    size = len(dataloader.dataset)
    class_accuracy, bbox_accuracy = 0., 0.

    with torch.no_grad():
        for X, y, z in dataloader:
            # Get predictions
            class_pred, bbox_pred = model(X)
            class_pred = class_pred.squeeze(1)

            # Calculate prediction accuracies
            class_accuracy += (torch.abs(y - class_pred) <= 0.5).sum().item()
            bbox_accuracy += (get_iou(z, bbox_pred)).sum().item()

    class_accuracy /= size
    bbox_accuracy /= size

    if writer != None and epoch != None:
        writer.add_scalar('class_accuracy', 100. * class_accuracy, epoch)
        writer.add_scalar('bbox_accuracy', 100. * bbox_accuracy, epoch)
    
    return class_accuracy, bbox_accuracy