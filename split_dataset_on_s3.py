import os
import random
import argparse
import boto3
from io import StringIO

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="train_ratio (default: 0.75)")
    parser.add_argument("--val_ratio", type=float, default=0.04,
                        help="val_ratio (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.01,
                        help="test_ratio (default: 0.05)")
    parser.add_argument("--s3_bucket", type=str, required=True,
                        help="S3 bucket name")
    parser.add_argument("--input_prefix", type=str, required=True,
                        help="S3 prefix for input data (containing samples and binary_masks)")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="S3 prefix for output data (where splits will be saved)")
    parser.add_argument("--from_csv", action='store_true', default=False,
                        help="this csv has metadata about the annotations, e.g, their SAM Score.")
    parser.add_argument("--csv_path", default=None, type=str,
                        help="S3 path to the CSV to read from.")
    parser.add_argument("--sam_threshold", type=float, default=0.95,
                        help="Mask will be used if the sam score > this value. (default: 0.95)")
    return parser

def list_s3_objects_resource(bucket_name, prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    return [os.path.basename(obj.key) for obj in bucket.objects.filter(Prefix=prefix) if not obj.key.endswith('/')]


def split_dataset_s3(s3_client, bucket, input_prefix, output_prefix, train_ratio, val_ratio, test_ratio):
    # List all objects in the samples directory
    print(f"Listing images from s3://{bucket}/{input_prefix}/samples/")
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{input_prefix}/samples/"
    )
    
    # Extract just the filenames from the full paths
    images = list_s3_objects_resource(bucket, f"{input_prefix}/samples/")
    print(f"Found {len(images)} images in S3 bucket")

    
    # Shuffle and split the dataset
    total_images = len(images)
    random.shuffle(images)
    
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    
    print(f"Split dataset: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Create and upload split files
    for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        # Create a string buffer to hold the file content
        buffer = StringIO()
        for item in split_images:
            # Remove extension but preserve case
            file_name_without_ext = item.rsplit('.', 1)[0]
            buffer.write(f"{file_name_without_ext}\n")
        
        # Reset buffer position to start
        buffer.seek(0)
        
        # Upload the file to S3
        s3_path = f"{output_prefix}/splits/{split_name}.txt"
        print(f"Uploading {split_name}.txt to s3://{bucket}/{s3_path}")
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_path,
            Body=buffer.getvalue()
        )
    
    return train_images, val_images, test_images

def main():
    # Parse arguments
    opts = get_argparser().parse_args()
    
    print("Starting S3 dataset splitting")
    print(f"Configuration: train_ratio={opts.train_ratio}, val_ratio={opts.val_ratio}, test_ratio={opts.test_ratio}")
    print(f"S3 bucket: {opts.s3_bucket}")
    print(f"Input prefix: {opts.input_prefix}")
    print(f"Output prefix: {opts.output_prefix}")
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Split dataset and upload to S3
    train_images, val_images, test_images = split_dataset_s3(
        s3_client=s3_client,
        bucket=opts.s3_bucket,
        input_prefix=opts.input_prefix,
        output_prefix=opts.output_prefix,
        train_ratio=opts.train_ratio,
        val_ratio=opts.val_ratio,
        test_ratio=opts.test_ratio
    )
    
    print(f"Dataset splitting completed successfully")
    print(f"Split files uploaded to s3://{opts.s3_bucket}/{opts.output_prefix}/splits/")

if __name__ == "__main__":
    main()