import boto3
import time
from datetime import datetime

# SageMaker client
sm_client = boto3.client('sagemaker')

# Configuration
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"sam-dino-psuedo-annotations-pipeline-{timestamp}"

# Replace with your values
account_id = ""
region = "eu-central-1"
role_arn = "arn:aws:iam::053061259712:role/service-role/AmazonSageMaker-ExecutionRole-"
image_uri = ".dkr.ecr.eu-central-1.amazonaws.com/segmentation:latest"
read_bucket_name = "treetracker-training-images"
write_bucket_name = 'sagemaker-segmentation-neel'
data_prefix = "crf_sam_annotations_large"
output_prefix = "sam2_dino_psuedo_annotation_pipeline/final_processed_file"

# Create training job
response = sm_client.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        'TrainingImage': image_uri,
        'TrainingInputMode': 'File',
    },
    RoleArn=role_arn,
    OutputDataConfig={
        'S3OutputPath': f"s3://{write_bucket_name}/{output_prefix}/"
    },
    ResourceConfig={
        'InstanceType': 'ml.g5.2xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 86400
    },
    CheckpointConfig={  # Fixed: Closing brace was missing
        'S3Uri': f's3://{write_bucket_name}/sam2_dino_psuedo_annotation_pipeline/temp_processed_file/',
        'LocalPath': '/opt/ml/checkpoints'
    },
    Environment={
    'DEVICE': 'cuda',
    'IMAGE_SIZE': '512,512',
    'SAM2_CHECKPOINT': 'checkpoints/sam2.1_hiera_base_plus.pt',
    'SAM2_MODEL_CFG': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'DINO_MODEL_ID': 'IDEA-Research/grounding-dino-tiny',
    
    'TEXT_QUERY': 'Plant',
    'DINO_BOX_THRESHOLD': '0.4',
    'DINO_TEXT_THRESHOLD': '0.3',
    'SAM_SCORE_THRESHOLD': '0.9',

    'S3_BUCKET': 'treetracker-training-images',
    'S3_SAMPLES_PREFIX': 'production_psuedo_labelling_100000/samples/',
    'S3_MASKS_PREFIX': 'production_psuedo_labelling_100000/binary_masks/',

    'MAX_PREFETCH_BATCHES': '3',
    'DOWNLOAD_WORKERS': '8',
    'TEMP_DIR': 'temp_downloads'
}
)

print(f"Training job '{job_name}' created.")
print("Waiting for training job to complete...")

# Wait for the training job to complete
while True:
    status = sm_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(f"Job status: {status}")
    
    if status in ['Completed', 'Failed', 'Stopped']:
        break
    
    time.sleep(60)  # Check every minute

print(f"Job {job_name} finished with status: {status}")