import boto3
from PIL import Image, ExifTags
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_exif_orientation(image_bytes):
    """Extract orientation value from EXIF data using the exact specified method"""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            # Exact method as specified
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = img._getexif()  # Using _getexif() to match the specified method
            
            if exif is not None and orientation in exif:
                return exif[orientation]
    except Exception as e:
        return None
    return None

def verify_image_mask_pairs_with_exif(bucket_name, sample_prefix, mask_prefix, max_workers=10):
    """
    Verify image-mask pairs and check for specific EXIF orientations (3,6,8).
    
    Args:
        bucket_name: S3 bucket name
        sample_prefix: Prefix for sample images
        mask_prefix: Prefix for mask images
        max_workers: Number of parallel threads
    """
    s3_client = boto3.client('s3')
    
    print("Listing all sample images...")
    sample_keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=sample_prefix):
        if 'Contents' in page:
            sample_keys.extend([obj['Key'] for obj in page['Contents']])
    
    print(f"Found {len(sample_keys)} sample images. Starting verification...")
    
    # Track images with specific orientations
    oriented_images = {
        3: [],
        6: [],
        8: []
    }
    
    def process_image(sample_key):
        # Check for mask existence first
        base_name = sample_key[len(sample_prefix):]
        mask_suffix = base_name[:-4] + '_binarymask.jpg' if base_name.endswith('.jpg') else base_name + '_binarymask.jpg'
        mask_key = mask_prefix + mask_suffix
        
        try:
            s3_client.head_object(Bucket=bucket_name, Key=mask_key)
            mask_exists = True
        except:
            mask_exists = False
        
        # Now check EXIF orientation using the specified method
        try:
            image_data = s3_client.get_object(Bucket=bucket_name, Key=sample_key)['Body'].read()
            orientation = get_exif_orientation(image_data)
            if orientation in [3, 6, 8]:
                return (sample_key, orientation, mask_exists)
        except:
            pass
        
        return None
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_image, sample_keys), total=len(sample_keys), desc="Processing images"))
    
    # Organize results
    missing_masks = []
    for result in results:
        if result is None:
            continue
        
        sample_key, orientation, mask_exists = result
        oriented_images[orientation].append(sample_key)
        if not mask_exists:
            missing_masks.append(sample_key)
    
    # Print orientation results
    print("\nEXIF Orientation Summary:")
    for orientation in [3, 6, 8]:
        count = len(oriented_images[orientation])
        print(f"Images with orientation {orientation}: {count}")
        if count > 0:
            print(f"First 5 examples with orientation {orientation}:")
            for sample_path in oriented_images[orientation][:5]:
                print(f"  - {sample_path}")
    
    # Print mask verification results
    print("\nMask Verification Summary:")
    print(f"Total samples: {len(sample_keys)}")
    print(f"Samples with missing masks: {len(missing_masks)}")
    
    if missing_masks:
        print("\nFirst 10 missing masks (sample paths):")
        for sample_path in missing_masks[:10]:
            print(sample_path)

if __name__ == "__main__":
    bucket_name = "treetracker-training-images"
    sample_prefix = "production_psuedo_labelling_100000/samples/"
    mask_prefix = "production_psuedo_labelling_100000/binary_masks/"
    
    verify_image_mask_pairs_with_exif(bucket_name, sample_prefix, mask_prefix)