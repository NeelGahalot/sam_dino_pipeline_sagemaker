# script to verify if we have all the samples and masks in the directory we wouls use for training the sagemaker job.

# Note that using a s3 resource would be faster than this paginnator method.

import boto3
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # for progress bar

def verify_image_mask_pairs(bucket_name, sample_prefix, mask_prefix, max_workers=10):
    """
    Verify that for every image in samples path, there's a corresponding mask in masks path.
    
    Args:
        bucket_name: S3 bucket name
        sample_prefix: Prefix for sample images (e.g., 'production_psuedo_labelling_100000/samples/')
        mask_prefix: Prefix for mask images (e.g., 'production_psuedo_labelling_100000/binary_masks/')
        max_workers: Number of parallel threads for verification
    """
    s3_client = boto3.client('s3')
    
    # First get all sample paths
    print("Listing all sample images...")
    sample_keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=sample_prefix):
        if 'Contents' in page:
            sample_keys.extend([obj['Key'] for obj in page['Contents']])
    
    print(f"Found {len(sample_keys)} sample images. Starting verification...")
    
    # Function to check individual pairs
    def verify_pair(sample_key):
        # Extract the base filename (without samples prefix)
        base_name = sample_key[len(sample_prefix):]
        
        # Remove .jpg extension and add _binarymask.jpg
        if base_name.endswith('.jpg'):
            mask_suffix = base_name[:-4] + '_binarymask.jpg'
        else:
            # Handle other extensions if needed
            mask_suffix = base_name + '_binarymask.jpg'
        
        mask_key = mask_prefix + mask_suffix
        
        # Check if mask exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=mask_key)
            return None  # No error means mask exists
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return sample_key  # Return the sample key if mask is missing
            else:
                raise  # Re-raise other errors
    
    # Process verification in parallel
    missing_masks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(verify_pair, sample_keys), total=len(sample_keys)))
    
    # Collect all missing masks
    missing_masks = [result for result in results if result is not None]
    
    # Print summary
    print("\nVerification complete!")
    print(f"Total samples: {len(sample_keys)}")
    print(f"Samples with missing masks: {len(missing_masks)}")
    
    if missing_masks:
        print("\nFirst 10 missing masks (sample paths):")
        for sample_path in missing_masks[:10]:
            print(sample_path)
        
        # Option to save full list to file
        save_file = input("\nSave full list to file? (y/n): ").lower()
        if save_file == 'y':
            with open('missing_masks.txt', 'w') as f:
                f.write("\n".join(missing_masks))
            print("Saved to missing_masks.txt")
    else:
        print("All samples have corresponding masks!")

# Example usage
if __name__ == "__main__":
    bucket_name = "treetracker-training-images"
    sample_prefix = "production_psuedo_labelling_100000/samples/"
    mask_prefix = "production_psuedo_labelling_100000/binary_masks/"
    
    verify_image_mask_pairs(bucket_name, sample_prefix, mask_prefix)