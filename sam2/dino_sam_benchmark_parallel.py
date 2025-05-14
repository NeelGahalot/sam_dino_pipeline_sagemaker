import os
import time
import numpy as np
import torch
import gc
from PIL import Image
from queue import Queue
from threading import Thread
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import SAM2 modules
from autosam_utils import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import DINO modules
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Constants
HOME = '/teamspace/studios/this_studio'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TEST_SIZE = 64  # Maximum batch size to try
TEST_IMAGE_SIZE = (512, 512)  # Target image size for benchmarking
SAM2_CHECKPOINT = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
NUM_RUNS = 3  # Number of test runs for each batch size for averaging
NUM_WARMUP = 2  # Number of warmup runs before timing

def benchmark_dino(max_batch_size=64):
    """Benchmark DINO independently with OOM error handling"""
    dino_times = []
    test_image = prepare_fixed_image()
    dino_model,dino_processor = load_dino_model()
    max_successful_batch = 0
    batch_sizes = [2**i for i in range(0, int(np.log2(max_batch_size)) + 1)]
    for batch_size in batch_sizes:
        print(f"\nTesting DINO batch size: {batch_size}")
        
        try:
            # Prepare batch
            img_batch = [test_image] * batch_size
            text_queries = [["object"]] * batch_size
            
            # Warm-up runs
            print("  Warming up DINO...")
            for _ in range(NUM_WARMUP):
                clear_gpu_memory()
                inputs = dino_processor(
                    text=text_queries,
                    images=img_batch,
                    return_tensors="pt"
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
            
            # Timing runs
            times = []
            for run in range(NUM_RUNS):
                print(f"  Run {run+1}/{NUM_RUNS}...")
                clear_gpu_memory()
                start_time = time.time()
                inputs = dino_processor(
                    text=text_queries,
                    images=img_batch,
                    return_tensors="pt"
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            dino_times.append(avg_time)
            max_successful_batch = batch_size
            print(f"  DINO avg time: {avg_time:.4f}s ({batch_size/avg_time:.2f} imgs/sec)")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  Batch size {batch_size}: Failed (CUDA OOM) ✗")
                # Pad results with NaN for failed batch sizes
                dino_times.extend([float('nan')] * (len(batch_sizes) - len(dino_times)))
                break
            else:
                print(f"  Batch size {batch_size}: Failed with error: {e} ✗")
                dino_times.extend([float('nan')] * (len(batch_sizes) - len(dino_times)))
                break
    
    print(f"\nMaximum successful DINO batch size: {max_successful_batch}")
    return dino_times

def benchmark_sam2_batch_size():
    """Find the maximum batch size that can be processed by SAM2."""
    results = BenchmarkResults()
    
    # Prepare a test image and bounding box
    test_image = prepare_fixed_image()
    test_box = generate_test_box(test_image.shape)
    
    print("Testing maximum batch size for SAM2...")
    batch_sizes = [2**i for i in range(0, int(np.log2(MAX_TEST_SIZE)) + 1)]
    # Try increasing batch sizes
    for batch_size in batch_sizes:
        try:
            clear_gpu_memory()
            predictor = load_sam2_model()
            
            # Create a batch of identical images and boxes
            img_batch = [test_image] * batch_size
            boxes_batch = [test_box] * batch_size
            
            # Try to process the batch
            predictor.set_image_batch(img_batch)
            masks_batch, scores_batch, _ = predictor.predict_batch(
                None, None, box_batch=boxes_batch, multimask_output=False
            )
            
            print(f"Batch size {batch_size}: Success ✓")
            results.max_batch_size = batch_size
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size {batch_size}: Failed (CUDA OOM) ✗")
                break
            else:
                print(f"Batch size {batch_size}: Failed with error: {e} ✗")
                break
    
    print(f"\nMaximum SAM2 batch size: {results.max_batch_size}")
    return results

def prepare_fixed_image(target_size=TEST_IMAGE_SIZE):
    """Create a fixed test image of the specified size."""
    # Create a simple test image or load and resize an existing one
    try:
        # Try to load an existing image and resize it
        image = Image.open(f'{HOME}/sam2/notebooks/images/truck.jpg')
        image = image.convert("RGB").resize(target_size)
    except:
        # Create a test pattern if no image is available
        image = Image.new("RGB", target_size, color=(128, 128, 128))
    
    return np.array(image)

def generate_test_box(image_shape):
    """Generate a sample bounding box for testing."""
    h, w = image_shape[:2]
    # Create a box that covers approximately the center 50% of the image
    x1 = int(w * 0.25)
    y1 = int(h * 0.25)
    x2 = int(w * 0.75)
    y2 = int(h * 0.75)
    return np.array([[x1, y1, x2, y2]])

def clear_gpu_memory():
    """Clear GPU memory between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_sam2_model():
    """Load the SAM2 model."""
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def load_dino_model():
    """Load the DINO model."""
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    return model, processor



import time
import threading
import queue
import numpy as np
import torch
import gc
from dataclasses import dataclass


@dataclass
class BenchmarkResults:
    """Container for benchmark results"""
    dino_throughput: float = 0.0
    sam_throughput: float = 0.0
    combined_throughput: float = 0.0
    optimal_dino_batch: int = 0
    optimal_sam_batch: int = 0
    max_queue_size: int = 0
    
def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class ParallelPipeline:
    """Pipeline to run DINO and SAM in parallel using queues"""
    def __init__(self, dino_batch_size, sam_batch_size, queue_size=50):
        self.dino_model, self.dino_processor = load_dino_model()
        self.sam_predictor = load_sam2_model()
        
        self.dino_batch_size = dino_batch_size
        self.sam_batch_size = sam_batch_size
        
        # Queue for passing DINO results to SAM
        self.detection_queue = queue.Queue(maxsize=queue_size)
        
        # Track processing metrics
        self.dino_processed = 0
        self.sam_processed = 0
        self.running = False
        self.dino_thread = None
        self.sam_thread = None
        
        # For measuring actual queue utilization
        self.max_queue_used = 0
        
    def dino_worker(self, images):
        """Worker thread for DINO processing"""
        remaining_images = images.copy()
        
        while self.running and remaining_images:
            # Get a batch of images to process
            batch_images = remaining_images[:self.dino_batch_size]
            remaining_images = remaining_images[self.dino_batch_size:]
            
            # Process with DINO
            text_queries = [["object"]] * len(batch_images)
            inputs = self.dino_processor(
                text=text_queries,
                images=batch_images,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            # Extract predicted boxes (simulated here)
            boxes = [generate_test_box(img.shape) for img in batch_images]
            
            # Put results in queue for SAM
            for img, box in zip(batch_images, boxes):
                self.detection_queue.put((img, box))
                self.dino_processed += 1
            
            # Track max queue usage
            self.max_queue_used = max(self.max_queue_used, self.detection_queue.qsize())
        
        # Signal that no more items will be produced
        self.detection_queue.put(None)
    
    def sam_worker(self):
        """Worker thread for SAM processing"""
        while self.running:
            batch_items = []
            
            # Try to fill a batch
            for _ in range(self.sam_batch_size):
                try:
                    item = self.detection_queue.get(timeout=0.1)
                    if item is None:  # End signal
                        self.running = False
                        break
                    batch_items.append(item)
                except queue.Empty:
                    break
            
            if not batch_items:
                continue
                
            # Process the batch with SAM
            img_batch = [item[0] for item in batch_items]
            boxes_batch = [item[1] for item in batch_items]
            
            self.sam_predictor.set_image_batch(img_batch)
            masks_batch, scores_batch, _ = self.sam_predictor.predict_batch(
                None, None, box_batch=boxes_batch, multimask_output=False
            )
            
            # Update counters
            self.sam_processed += len(batch_items)
            
            # Mark items as done in the queue
            for _ in range(len(batch_items)):
                self.detection_queue.task_done()
    
    def process(self, images):
        """Process a set of images through the pipeline"""
        self.running = True
        self.dino_processed = 0
        self.sam_processed = 0
        self.max_queue_used = 0
        
        # Start worker threads
        self.dino_thread = threading.Thread(target=self.dino_worker, args=(images,))
        self.sam_thread = threading.Thread(target=self.sam_worker)
        
        start_time = time.time()
        
        self.dino_thread.start()
        self.sam_thread.start()
        
        # Wait for completion
        self.dino_thread.join()
        self.sam_thread.join()
        
        total_time = time.time() - start_time
        
        return {
            "dino_processed": self.dino_processed,
            "sam_processed": self.sam_processed,
            "total_time": total_time,
            "max_queue_used": self.max_queue_used
        }

def benchmark_parallel_pipeline(dino_batch_size, sam_batch_size, num_images=1000, queue_size=50):
    """Benchmark the parallel pipeline with specific batch sizes"""
    print(f"\nBenchmarking pipeline with DINO batch={dino_batch_size}, SAM batch={sam_batch_size}")
    
    # Prepare test images
    test_image = prepare_fixed_image()
    images = [test_image] * num_images
    
    # Create pipeline
    pipeline = ParallelPipeline(dino_batch_size, sam_batch_size, queue_size)
    
    # Warm-up run
    print("  Warming up pipeline...")
    warm_up_images = [test_image] * min(100, num_images)
    clear_gpu_memory()
    pipeline.process(warm_up_images)
    
    # Actual benchmark run
    print(f"  Processing {num_images} images...")
    clear_gpu_memory()
    results = pipeline.process(images)
    
    # Calculate metrics
    throughput = results["sam_processed"] / results["total_time"]
    
    print(f"  Results:")
    print(f"    Total time: {results['total_time']:.2f} seconds")
    print(f"    DINO images processed: {results['dino_processed']}")
    print(f"    SAM images processed: {results['sam_processed']}")
    print(f"    Overall throughput: {throughput:.2f} images/sec")
    print(f"    Max queue utilization: {results['max_queue_used']}/{queue_size}")
    
    return {
        "dino_batch": dino_batch_size,
        "sam_batch": sam_batch_size,
        "throughput": throughput,
        "max_queue_used": results["max_queue_used"]
    }

def find_optimal_configuration(max_dino_batch=64, max_sam_batch=64, num_images=1000):
    """Find the optimal batch sizes for DINO and SAM in parallel processing"""
    results = BenchmarkResults()
    
    print("\n=== Finding Optimal DINO+SAM Configuration ===")
    
    # First find individual optimums
    print("\nFinding maximum batch sizes for individual models...")
    dino_times = benchmark_dino(max_batch_size=max_dino_batch)
    sam_results = benchmark_sam2_batch_size()
    
    # Calculate potential batch sizes to test
    dino_batch_sizes = [2**i for i in range(0, int(np.log2(max_dino_batch)) + 1)]
    sam_batch_sizes = [2**i for i in range(0, int(np.log2(max_sam_batch)) + 1)]
    
    valid_dino_batches = [b for i, b in enumerate(dino_batch_sizes) if not np.isnan(dino_times[i])]
    valid_sam_batches = [b for b in sam_batch_sizes if b <= sam_results.max_batch_size]
    
    if not valid_dino_batches or not valid_sam_batches:
        print("Error: Could not find valid batch sizes for either DINO or SAM")
        return results
    
    # Define combinations to test - focusing on powers of 2
    test_combinations = []
    for d in valid_dino_batches:
        for s in valid_sam_batches:
            # Only test reasonable combinations
            if 0.25 <= (d/s) <= 4:
                test_combinations.append((d, s))
    
    # Sort by total batch size (larger first)
    test_combinations.sort(key=lambda x: x[0] + x[1], reverse=True)
    
    # Limit to top combinations to save time
    test_combinations = test_combinations[:min(8, len(test_combinations))]
    
    # Test each combination
    best_throughput = 0
    best_config = None
    
    for dino_batch, sam_batch in test_combinations:
        # Try different queue sizes as well
        queue_size = max(50, dino_batch * 3)  # Enough to buffer several DINO batches
        
        result = benchmark_parallel_pipeline(
            dino_batch_size=dino_batch,
            sam_batch_size=sam_batch,
            num_images=num_images,
            queue_size=queue_size
        )
        
        if result["throughput"] > best_throughput:
            best_throughput = result["throughput"]
            best_config = result
    
    if best_config:
        results.optimal_dino_batch = best_config["dino_batch"]
        results.optimal_sam_batch = best_config["sam_batch"]
        results.combined_throughput = best_config["throughput"]
        results.max_queue_size = best_config["max_queue_used"]
        
    print(f"\n=== Optimal Configuration ===")
    print(f"DINO batch size: {results.optimal_dino_batch}")
    print(f"SAM batch size: {results.optimal_sam_batch}")
    print(f"Combined throughput: {results.combined_throughput:.2f} images/sec")
    print(f"Recommended queue size: {results.max_queue_size * 1.5:.0f}")
    
    # Calculate theoretical time for 100,000 images
    est_hours = 100000 / (results.combined_throughput * 3600)
    print(f"\nEstimated time for 100,000 images on current hardware: {est_hours:.2f} hours")
    
    return results

if __name__ == "__main__":
    # Run the optimization benchmark
    optimal_config = find_optimal_configuration(
        max_dino_batch=64,  # Adjust based on your GPU memory
        max_sam_batch=64,   # Adjust based on your GPU memory
        num_images=1000     # Number of images for test
    )