from torch import nn, save
from torchvision.models import mobilenet_v3_large

class FaceTracker(nn.Module):
    '''
    Custom PyTorch neural network based on the MobileNet V3 architecture.
    Splits into two heads for object classification and bounding box predictions.
    '''
    def __init__(self, cw: float, lw: float):
        super().__init__()
        self.cw = cw
        self.lw = lw
        self.body = mobilenet_v3_large().features
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 2048),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Linear(2048, 4)
    
    def forward(self, x):
        x = self.neck(self.body(x))
        class_pred = self.cw * self.classifier(x)
        bbox_pred = self.lw * self.regression(x)
        return class_pred, bbox_pred

if __name__ == '__main__':
    model_name = 'firecracker'
    save(FaceTracker().state_dict(), f'model_weights\\{model_name}.pth')