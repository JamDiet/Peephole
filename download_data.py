import subprocess
import zipfile
import os
from tqdm import tqdm

def download_data(output_dir: str='data'):
    # Retrieve data zip folder
    subprocess.run(['zenodo_get', '-r', '14474899'])
    zip_path = 'soloface-detection-dataset.zip'

    # Create data directory if nonexistant
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    root_prefix = 'soloface-detection-dataset/'

    with zipfile.ZipFile(zip_path) as z:
        print('\nExtracting files to output directory...')
        
        for member in tqdm(z.namelist()):
            if member != root_prefix:   # Avoid saving root folder
                relative_path = member[len(root_prefix):]

                if relative_path.startswith('README'):  # Avoid saving README
                    continue
                
                target_path = os.path.join(output_dir, relative_path)

                # Create subdirectory if nonexistant
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Write file to target path
                if not os.path.isdir(target_path):
                    with z.open(member) as src, open(target_path, 'wb') as dst:
                        dst.write(src.read())

    os.remove(zip_path)

if __name__ == '__main__':
    download_data(output_dir='data')