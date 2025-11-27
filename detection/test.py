import os
import uuid
from torch import load
from facetracker import FaceTracker
from utils import CustomImageDataset, random_annotation

def test(
        model_name: str,
        num_images: int,
        data_root: str='data'
):
    model = FaceTracker()

    # Load existing model weights
    model_path = os.path.join('detection', 'model_weights', f'{model_name}.pth')
    if os.path.exists(model_path):
        print('Loading existing model...\n')
        model.load_state_dict(load(model_path, weights_only=True))
    else:
        print(f'Cannot find {model_name}. Please provide another model or begin by training {model_name}.')
    
    testing_data = CustomImageDataset('test', data_root)

    for _ in range(num_images):
        random_annotation(
                    model,
                    testing_data,
                    os.path.join('detection', 'val_annotations', f'{uuid.uuid4()}.png')
                )
        
if __name__ == '__main__':
    test(
        model_name='base',
        num_images=2,
        data_root='data'
    )