import os
import uuid
import argparse
import yaml
from torch import load
from facetracker import FaceTracker
from utils import CustomImageDataset, random_annotation

def validate(
        model_name: str,
        num_images: int,
        data_root: str='data'
):
    model = FaceTracker()

    # Load existing model weights
    model_path = os.path.join('detection', 'model_weights', f'{model_name}.pth')
    if os.path.exists(model_path):
        print(f'Loading existing weights from {model_name}...\n')
        model.load_state_dict(load(model_path, weights_only=True))
    else:
        print(f'Cannot find {model_name}. Please provide another model or begin by training {model_name}.')
    
    testing_data = CustomImageDataset('val', data_root)

    for _ in range(num_images):
        random_annotation(
                    model,
                    testing_data,
                    os.path.join('detection', 'val_annotations', f'{uuid.uuid4()}.png')
                )
        
def main(args):
    validate(
        model_name=args.model_name,
        num_images=args.num_images,
        data_root=args.data_root
    )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images")
    parser.add_argument("--data_root", type=str, default="data", help="Top-level data directory")

    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(open(args.config))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    main(args)