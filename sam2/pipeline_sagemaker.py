import os
import time
import numpy as np
import torch
import pandas as pd
import requests
import io
import boto3
from PIL import Image, ImageDraw
from tqdm import tqdm
import concurrent.futures
from queue import Queue
from threading import Thread, Event
#import matplotlib.pyplot as plt
import csv
import gc
from urllib.parse import urlparse
from queue import Empty
import argparse

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import DINO modules
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import Tuple



def get_config_parser() -> argparse.ArgumentParser:
    """
    Creates and returns an argument parser with configuration options for the pipeline.
    All parameters can be set via environment variables or command-line arguments.
    Command-line arguments take precedence over environment variables.
    """
    HOME = os.getcwd()
    parser = argparse.ArgumentParser(
        description="Image Processing Pipeline for Plant Detection and Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--final_results_csv',
                            type=str,
                            default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
                            help='final csv (/opt/ml/model contents uploaded to S3 after training)')
    parser.add_argument('--temp_results_csv',
                            type=str,
                            default=os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'),
                            help='Temp csv, saved after 10 batches (/opt/ml/checkpointssynced with S3 during training)')
    parser.add_argument('--batch-size',
                            type=int,
                            default=os.environ.get('BATCH_SIZE', 16),
                            help='Directory for periodic checkpoints (synced with S3 during training)')

    # System and Hardware Configuration
    parser.add_argument('--device', 
                        type=str,
                        default=os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'),
                        choices=['cpu', 'cuda'],
                        help='Device to run models on (cpu/cuda)')

    # Image Processing Configuration
    parser.add_argument('--image-size',
                        type=lambda s: tuple(map(int, s.split(','))),
                        default=tuple(map(int, os.environ.get('IMAGE_SIZE', '512,512').split(','))),
                        help='Target image dimensions as "height,width"')

    # Model Paths and Configurations
    model_group = parser.add_argument_group('Model Configurations')
    model_group.add_argument('--sam2-checkpoint',
                            type=str,
                            default=os.environ.get('SAM2_CHECKPOINT', f"{HOME}/checkpoints/sam2.1_hiera_base_plus.pt"),
                            help='Path to SAM2 checkpoint file')
    model_group.add_argument('--sam2-model-cfg',
                            type=str,
                            default=os.environ.get('SAM2_MODEL_CFG', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
                            help='Path to SAM2 model config file')
    model_group.add_argument('--dino-model-id',
                            type=str,
                            default=os.environ.get('DINO_MODEL_ID', 'IDEA-Research/grounding-dino-tiny'),
                            help='HuggingFace model ID for GroundingDINO')

    # Detection Thresholds
    threshold_group = parser.add_argument_group('Detection Thresholds')
    threshold_group.add_argument('--text-query',
                               type=str,
                               default=os.environ.get('TEXT_QUERY', 'Plant'),
                               help='Text query for object detection')
    threshold_group.add_argument('--dino-box-threshold',
                               type=float,
                               default=float(os.environ.get('DINO_BOX_THRESHOLD', 0.4)),
                               help='DINO bounding box confidence threshold')
    threshold_group.add_argument('--dino-text-threshold',
                               type=float,
                               default=float(os.environ.get('DINO_TEXT_THRESHOLD', 0.3)),
                               help='DINO text similarity threshold')
    threshold_group.add_argument('--sam-score-threshold',
                               type=float,
                               default=float(os.environ.get('SAM_SCORE_THRESHOLD', 0.9)),
                               help='SAM mask prediction quality threshold')

    # S3 Storage Configuration
    s3_group = parser.add_argument_group('S3 Storage Configuration')
    s3_group.add_argument('--s3-bucket',
                         type=str,
                         default=os.environ.get('S3_BUCKET', 'treetracker-training-images'),
                         help='S3 bucket name for storing results')
    s3_group.add_argument('--s3-samples-prefix',
                         type=str,
                         default=os.environ.get('S3_SAMPLES_PREFIX', 'production_psuedo_labelling_100000/samples/'),
                         help='S3 prefix path for sample images')
    s3_group.add_argument('--s3-masks-prefix',
                         type=str,
                         default=os.environ.get('S3_MASKS_PREFIX', 'production_psuedo_labelling_100000/binary_masks/'),
                         help='S3 prefix path for binary masks')

    # Pipeline Performance Configuration
    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument('--max-prefetch-batches',
                          type=int,
                          default=int(os.environ.get('MAX_PREFETCH_BATCHES', 3)),
                          help='Maximum number of batches to prefetch')
    perf_group.add_argument('--download-workers',
                          type=int,
                          default=int(os.environ.get('DOWNLOAD_WORKERS', 8)),
                          help='Number of parallel download workers')
    perf_group.add_argument('--temp-dir',
                          type=str,
                          default=os.environ.get('TEMP_DIR', 'temp_downloads'),
                          help='Temporary directory for downloaded images')

    return parser




class AsyncImageLoader:
    """Asynchronously loads images from URLs and feeds them to the processing pipeline."""
    
    def __init__(self, csv_path, batch_size=8, max_prefetch=MAX_PREFETCH_BATCHES, num_workers=DOWNLOAD_WORKERS):
        """
        Initialize the async loader.
        
        Args:
            csv_path: Path to CSV file containing image URLs
            batch_size: Number of images per batch
            max_prefetch: Maximum number of batches to prefetch
            num_workers: Number of download workers
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.max_prefetch = max_prefetch
        self.num_workers = num_workers
        
        # Create temporary directory
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Load URLs from CSV
        self.urls_df = pd.read_csv(self.csv_path)
        if 'image_url' not in self.urls_df.columns:
            raise ValueError("CSV must contain 'image_url' column")
        
        self.total_images = len(self.urls_df)
        print(f"Found {self.total_images} image URLs in the CSV")
        
        # Queues for communication between threads
        self.download_queue = Queue(maxsize=max_prefetch * batch_size)
        self.batch_queue = Queue(maxsize=max_prefetch)
        
        # Control flags
        self.stop_event = Event()
        self.is_running = False
        
        # Batch tracking
        self.current_batch_idx = 0
        self.num_batches = (self.total_images + batch_size - 1) // batch_size
        
    def get_filename_from_url(self, url, idx):
        """Extract a valid filename from URL with fallback to index."""
        try:
            parsed_url = urlparse(url)
            basename = os.path.basename(parsed_url.path)
            
            # If no basename or invalid, use index
            if not basename or '.' not in basename:
                basename = f"image_{idx:08d}.jpg"
                
            return basename
        except:
            return f"image_{idx:08d}.jpg"
    
    def download_image(self, url, idx):
        """Download a single image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Get filename
            basename = self.get_filename_from_url(url, idx)
            save_path = os.path.join(TEMP_DIR, basename)
            
            # Save the image to disk
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
            # Create image metadata
            metadata = {
                'path': save_path,
                'url': url,
                'basename': basename,
                'idx': idx
            }
            
            return metadata
        except Exception as e:
            # Return None for failed downloads
            print(f"Error downloading {url}: {e}")
            return None
    
    def download_worker(self, worker_id):
        """Worker thread that downloads images."""
        while not self.stop_event.is_set():
            # Get next item from queue
            try:
                idx, url = self.download_queue.get(timeout=1)
                
                # Download the image
                metadata = self.download_image(url, idx)
                
                # Put result in batch building queue
                if metadata:
                    self.batch_queue.put(metadata)
                    
                # Mark task as done
                self.download_queue.task_done()
            except Empty:
                # No more items in queue
                continue
            except Exception as e:
                print(f"Error in download worker {worker_id}: {e}")
                self.download_queue.task_done()
    
    def batch_builder(self):
        """Build batches from downloaded images."""
        current_batch = []
        current_batch_idx = 0
        processed_count = 0
        results = []
        
        while not self.stop_event.is_set() or processed_count < self.total_images:
            try:
                # Get downloaded image metadata
                metadata = self.batch_queue.get(timeout=5)
                
                # Add to current batch
                current_batch.append(metadata)
                processed_count += 1
                
                # If batch is complete or last batch
                if len(current_batch) == self.batch_size or processed_count == self.total_images:
                    # Process batch
                    yield current_batch_idx, current_batch
                    
                    # Prepare for next batch
                    current_batch = []
                    current_batch_idx += 1
                    
                # Mark task as done
                self.batch_queue.task_done()
            except Empty:  # Changed from Queue.Empty to Empty
                # Check if all downloads are done
                if processed_count >= self.total_images:
                    break
                continue
            except Exception as e:
                print(f"Error in batch builder: {e}")
                self.batch_queue.task_done()
        
        print("Batch builder finished")
    
    def queue_downloader(self):
        """Queue URLs for downloading."""
        for idx, row in enumerate(self.urls_df.itertuples()):
            if self.stop_event.is_set():
                break
                
            url = getattr(row, 'image_url')
            self.download_queue.put((idx, url))
    
    def start(self):
        """Start the async loading pipeline."""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        # Start download workers
        self.download_threads = []
        for i in range(self.num_workers):
            thread = Thread(target=self.download_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.download_threads.append(thread)
        
        # Queue downloads
        self.queue_thread = Thread(target=self.queue_downloader)
        self.queue_thread.daemon = True
        self.queue_thread.start()
        
        # Return the batch generator
        return self.batch_builder()
    
    def stop(self):
        """Stop the async loading pipeline."""
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.download_threads:
            thread.join(timeout=1)
            
        self.queue_thread.join(timeout=1)
        self.is_running = False

def load_models():
    """Load DINO and SAM2 models."""
    print("Loading DINO model...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    
    print("Loading SAM2 model...")
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    return dino_model, dino_processor, sam2_predictor

def clear_gpu_memory():
    """Clear GPU memory between processing batches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_s3_client():
    """Create and return a boto3 S3 client."""
    return boto3.client('s3')

def upload_to_s3(s3_client, file_path, s3_key):
    """Upload a file to S3 bucket."""
    try:
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False

def save_binary_mask(mask, save_path):
    """Save a binary mask as an image."""
    # Convert mask to binary image (0 or 255)
    binary_mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(binary_mask)
    mask_img.save(save_path)
    return save_path

def load_and_preprocess_image(metadata):
    """Load and preprocess a single image."""
    path = metadata['path']
    try:
        image = Image.open(path)
        image = image.convert("RGB").resize(IMAGE_SIZE)
        return np.array(image), metadata
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # Return a gray placeholder image if loading fails
        placeholder = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8) * 128
        return placeholder, metadata

def process_images(csv_path, results_dir, results_csv, batch_size=8):
    """
    Main processing function.
    
    Args:
        csv_path: Path to CSV file containing image URLs
        results_dir: Directory to save results
        results_csv: Path to save results CSV
        batch_size: Batch size for processing
    """
    # Create results directories
    masks_dir = os.path.join(results_dir, "masks")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Step 1: Create async image loader
    loader = AsyncImageLoader(csv_path, batch_size=batch_size)
    
    # Step 2: Load models
    dino_model, dino_processor, sam2_predictor = load_models()
    
    # Step 3: Create S3 client
    s3_client = create_s3_client()
    
    # Step 4: Initialize results list
    results = []
    
    # Start the loader
    batch_generator = loader.start()
    
    try:
        # Process batches as they become available
        for batch_idx, batch_metadata in tqdm(batch_generator, desc="Processing batches", total=loader.num_batches):
            if not batch_metadata:
                continue
                
            # Load and preprocess images
            batch_data = [load_and_preprocess_image(metadata) for metadata in batch_metadata]
            batch_images = [img for img, _ in batch_data]
            batch_meta = [meta for _, meta in batch_data]
            
            # Skip empty batch
            if not batch_images:
                continue
                
            # Process with DINO
            text_queries = [[TEXT_QUERY]] * len(batch_images)  # Can be replaced with your specific query
            inputs = dino_processor(
                text=text_queries,
                images=batch_images,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
            
            # Process results to get bounding boxes
            batch_sizes = [(img.shape[0], img.shape[1]) for img in batch_images]
            dino_results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=DINO_BOX_THRESHOLD,
                text_threshold=DINO_TEXT_THRESHOLD,
                target_sizes=batch_sizes
            )
            
            # Process each image in the batch
            for j, (image, meta, result) in enumerate(zip(batch_images, batch_meta, dino_results)):
                path = meta['path']
                basename = meta['basename']
                url = meta['url']
                basename_no_ext = os.path.splitext(basename)[0]
                
                result_entry = {
                    'image_url': url,
                    'image_path': path,
                    'basename': basename,
                    'dino_detected': len(result["boxes"]) > 0,
                    'dino_score': None,
                    'sam_score': None,
                    'passed_thresholds': False,
                    'uploaded_to_s3': False,
                    'box_area_pct': None,

                }
                
                if len(result["boxes"]) == 0:
                    results.append(result_entry)
                    # Clean up temporary image
                    try:
                        os.remove(path)
                    except:
                        pass
                    continue
                
                # Get only the first (highest confidence) detection
                box = result["boxes"][0].tolist()
                x0, y0, x1, y1 = box
                box_area = (x1 - x0) * (y1 - y0)
                image_area = image.shape[0] * image.shape[1]
                box_area_pct = box_area / image_area
                dino_score = result["scores"][0].item()
                result_entry['dino_score'] = dino_score
                result_entry['box_area_pct'] = box_area_pct
                
                # Process with SAM2 if DINO score meets threshold
                if dino_score >= DINO_BOX_THRESHOLD and box_area_pct < 0.8:
                    # Prepare box for SAM
                    sam_box = np.array([[box]])
                    
                    # Process with SAM2
                    sam2_predictor.set_image_batch([image])
                    masks_batch, scores_batch, _ = sam2_predictor.predict_batch(
                        None, None, box_batch=[sam_box], multimask_output=False
                    )
                    
                    # Get the mask and score
                    mask = masks_batch[0][0]
                    sam_score = scores_batch[0][0]
                    result_entry['sam_score'] = sam_score.item()
                    
                    # Check if SAM score meets threshold
                    if sam_score >= SAM_SCORE_THRESHOLD:
                        result_entry['passed_thresholds'] = True
                        
                        # Save the image with bounding box
                        img_with_box = Image.fromarray(image.copy())
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=2)
                        result_image_path = os.path.join(results_dir, basename)
                        img_with_box.save(result_image_path)
                        
                        # Save the binary mask
                        mask_path = os.path.join(masks_dir, f"{basename_no_ext}_binarymask.jpg")
                        save_binary_mask(mask, mask_path)
                        
                        # Upload to S3
                        try:
                            s3_sample_key = f"{S3_SAMPLES_PREFIX}{basename}"
                            s3_mask_key = f"{S3_MASKS_PREFIX}{basename_no_ext}_binarymask.jpg"
                            
                            upload_to_s3(s3_client, result_image_path, s3_sample_key)
                            upload_to_s3(s3_client, mask_path, s3_mask_key)
                            result_entry['uploaded_to_s3'] = True
                        except Exception as e:
                            print(f"Error uploading to S3: {e}")
                
                results.append(result_entry)
                
                # Clean up temporary image
                try:
                    os.remove(path)
                except:
                    pass
            
            # Save intermediate results every 10 batches 
            if batch_idx % 10 == 0 and batch_idx > 0:
                # Save intermediate results to CSV
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(results_csv, index=False)
                print(f"Saved intermediate results ({len(results)} images processed so far)")
                try:
                    for file in os.listdir(TEMP_DIR):
                        os.remove(os.path.join(TEMP_DIR, file))
                        
                    for file in os.listdir(masks_dir):
                        os.remove(os.path.join(masks_dir, file))
                    
                except:
                    pass
            
            # Clear GPU memory after each batch
            clear_gpu_memory()
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Stop the loader
        loader.stop()
        
        # Save final results to CSV
        print(f"Saving final results to {results_csv}...")
        df = pd.DataFrame(results)
        df.to_csv(final_results_csv, index=False)
        
        # Clean up temporary directory
        try:
            for file in os.listdir(TEMP_DIR):
                os.remove(os.path.join(TEMP_DIR, file))
        except:
            pass
            
        print("Processing complete!")

def main():
    # Configuration
    parser = get_config_parser()
    args = parser.parse_args()
    
    csv_path = "big_production_sampling.csv"  # Path to CSV with image URLs
    results_dir = "processing_results"  # Directory to save results
    
    results_csv = f"{args.temp_results_csv}/processing_results.csv"  # Path to save results CSV
    final_results_csv = f"{args.final_results_csv}/processing_results.csv"
    batch_size = args.batch_size  # Adjust based on available GPU memory

    HOME = os.getcwd()
    DEVICE = args.device
    IMAGE_SIZE = args.image_size

    # Model Paths and Configurations
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CFG = args.sam2_model_cfg
    DINO_MODEL_ID = args.dino_model_id

    # Processing thresholds
    TEXT_QUERY = args.text_query
    DINO_BOX_THRESHOLD = args.dino_box_threshold
    DINO_TEXT_THRESHOLD = args.dino_text_threshold
    SAM_SCORE_THRESHOLD = args.sam_score_threshold

    # S3 bucket configuration
    S3_BUCKET = args.s3_bucket
    S3_SAMPLES_PREFIX = args.s3_samples_prefix
    S3_MASKS_PREFIX = args.s3_masks_prefix

    # Async pipeline configuration
    MAX_PREFETCH_BATCHES = args.max_prefetch_batches
    DOWNLOAD_WORKERS = args.download_workers
    TEMP_DIR = args.temp_dir
    
    # Process images
    process_images(csv_path, results_dir, results_csv, batch_size)

if __name__ == "__main__":
    main()