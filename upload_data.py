import boto3
import os
import argparse

def upload_face_data(bucket, data_root: str):
    s3 = boto3.client()

    for partition in ["train", "test", "val"]:
        data_dir = os.path.join(data_root, partition)
        label_dir = os.path.join(data_dir, 'labels')
        img_labels = os.listdir(label_dir)
        img_dir = os.path.join(data_dir, 'images')

        for idx in img_labels:
            img_path = os.path.join(img_dir, img_labels[idx].split('.json')[0] + '.jpg')
            s3.upload_file(img_path, bucket)

            label_path = os.path.join(label_dir, img_labels[idx])
            s3.upload_file(label_path, bucket)

def main(args):
    upload_face_data(args.bucket, args.data_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", type=str, help="Name of AWS S3 bucket")
    parser.add_argument("--data_root", type=str, default="data", help="Top-level data directory")

    args = parser.parse_args()
    main(args)