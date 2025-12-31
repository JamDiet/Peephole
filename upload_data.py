import boto3
import os
import argparse
import json

def upload_face_data(
        bucket,
        data_root: str,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str
):
    s3 = boto3.client(
        "s3",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    for partition in ["train", "test", "val"]:
        data_dir = os.path.join(data_root, partition)
        label_dir = os.path.join(data_dir, 'labels')
        img_labels = os.listdir(label_dir)
        img_dir = os.path.join(data_dir, 'images')

        for idx in range(len(img_labels)):
            img_path = os.path.join(img_dir, img_labels[idx].split('.json')[0] + '.jpg')
            s3.upload_file(img_path, bucket, img_path)

            label_path = os.path.join(label_dir, img_labels[idx])
            s3.upload_file(label_path, bucket, label_path)

def main(args):
    upload_face_data(
        bucket=args.bucket,
        data_root=args.data_root,
        region_name=args.region_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", type=str, help="Name of AWS S3 bucket")
    parser.add_argument("--data_root", type=str, default="data", help="Top-level data directory")
    parser.add_argument("--region_name", type=str, help="AWS bucket region")
    parser.add_argument("--aws_access_key_id", type=str, help="AWS access key ID")
    parser.add_argument("--aws_secret_access_key", type=str, help="AWS secret access key")
    parser.add_argument("--config_data", type=str, help="AWS access credentials")

    args = parser.parse_args()

    if args.config_data:
        with open(args.config_data, "r") as f:
            config_data = json.load(f)
        for k, v in config_data.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    main(args)