from torch import nn, save
from torchvision.models import mobilenet_v3_large, vgg16

class FaceTracker(nn.Module):
    '''
    Custom PyTorch neural network based on the MobileNet V3 architecture.
    Splits into two heads for object classification and bounding box predictions.
    '''
    def __init__(self):
        super().__init__()
        self.body = vgg16().features
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4)
        )
    
    def forward(self, x):
        x = self.neck(self.body(x))
        class_pred = self.classifier(x)
        bbox_pred = self.regression(x)
        return class_pred, bbox_pred