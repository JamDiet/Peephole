# Peephole

This repository will contain all the code necessary to train and run a single-face detection and recognition model. I personally intend to use it for unlocking my dormitory door with my face, but you may do with it what you will.

## Downloading Data

The data used to train the face detection model can be downloaded to your current working directory by running the following in the terminal:
```bash
python download_data.py --output_dir="data"
```
A folder called "data" will be added to your CWD with "train", "test", and "val" folders containing the corresponding images and labels. Size: 1.2 GB. <br> <br>
For more information about the dataset or to download it manually, please visit https://zenodo.org/records/14474899.
