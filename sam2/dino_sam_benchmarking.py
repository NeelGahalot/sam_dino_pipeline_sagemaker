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

class BenchmarkResults:
    def __init__(self):
        self.batch_sizes = []
        self.sam_times = []
        self.dino_times = []
        self.combined_times = []
        self.max_batch_size = 0
        
    def plot(self, save_path="benchmark_results.png"):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.batch_sizes, self.sam_times, 'b-o', label='SAM2')
        plt.plot(self.batch_sizes, self.dino_times, 'r-o', label='DINO')
        plt.plot(self.batch_sizes, self.combined_times, 'g-o', label='Combined')
        plt.title('Processing Time vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        # Calculate throughput (images/second)
        sam_throughput = [bs/t if t > 0 else 0 for bs, t in zip(self.batch_sizes, self.sam_times)]
        dino_throughput = [bs/t if t > 0 else 0 for bs, t in zip(self.batch_sizes, self.dino_times)]
        combined_throughput = [bs/t if t > 0 else 0 for bs, t in zip(self.batch_sizes, self.combined_times)]
        
        plt.plot(self.batch_sizes, sam_throughput, 'b-o', label='SAM2')
        plt.plot(self.batch_sizes, dino_throughput, 'r-o', label='DINO')
        plt.plot(self.batch_sizes, combined_throughput, 'g-o', label='Combined')
        plt.title('Throughput vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (images/second)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

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

def benchmark_processing_time(max_batch_size):
    """Benchmark processing time for different batch sizes."""
    results = BenchmarkResults()
    test_image = prepare_fixed_image()
    test_box = generate_test_box(test_image.shape)
    
    # For DINO
    dino_model, dino_processor = load_dino_model()
    
    # Determine batch sizes to test (powers of 2 up to max_batch_size)
    batch_sizes = [2**i for i in range(0, int(np.log2(max_batch_size)) + 1)]
    if max_batch_size not in batch_sizes:
        batch_sizes.append(max_batch_size)
    batch_sizes.sort()
    
    results.batch_sizes = batch_sizes
    print("\nBenchmarking processing time for different batch sizes...")
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Prepare batches
        img_batch = [test_image] * batch_size
        boxes_batch = [test_box] * batch_size
        
        # Prepare text queries for DINO
        text_queries = [["object"]] * batch_size
        
        # Warm-up runs
        print("  Warming up...")
        for _ in range(NUM_WARMUP):
            # Warm-up SAM2
            clear_gpu_memory()
            sam2_predictor = load_sam2_model()
            sam2_predictor.set_image_batch(img_batch)
            sam2_predictor.predict_batch(None, None, box_batch=boxes_batch, multimask_output=False)
            
            # Warm-up DINO
            inputs = dino_processor(
                text=text_queries,
                images=img_batch,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
            
        # Actual timing runs
        sam_times = []
        dino_times = []
        combined_times = []
        
        for run in range(NUM_RUNS):
            print(f"  Run {run+1}/{NUM_RUNS}...")
            
            # Time SAM2
            clear_gpu_memory()
            sam2_predictor = load_sam2_model()
            start_time = time.time()
            sam2_predictor.set_image_batch(img_batch)
            masks_batch, scores_batch, _ = sam2_predictor.predict_batch(
                None, None, box_batch=boxes_batch, multimask_output=False
            )
            sam_time = time.time() - start_time
            sam_times.append(sam_time)
            
            # Time DINO
            clear_gpu_memory()
            start_time = time.time()
            inputs = dino_processor(
                text=text_queries,
                images=img_batch,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
            dino_time = time.time() - start_time
            dino_times.append(dino_time)
            
            # Combined time (worst case if they had to run sequentially)
            combined_times.append(sam_time + dino_time)
        
        # Calculate average times
        avg_sam_time = np.mean(sam_times)
        avg_dino_time = np.mean(dino_times)
        avg_combined_time = np.mean(combined_times)
        
        results.sam_times.append(avg_sam_time)
        results.dino_times.append(avg_dino_time)
        results.combined_times.append(avg_combined_time)
        
        print(f"  SAM2 avg time: {avg_sam_time:.4f}s ({batch_size/avg_sam_time:.2f} imgs/sec)")
        print(f"  DINO avg time: {avg_dino_time:.4f}s ({batch_size/avg_dino_time:.2f} imgs/sec)")
        print(f"  Combined avg time: {avg_combined_time:.4f}s ({batch_size/avg_combined_time:.2f} imgs/sec)")
    
    return results

class ParallelPipeline:
    """Parallel processing pipeline for DINO and SAM2."""
    def __init__(self, batch_size, dino_model, dino_processor, sam2_predictor):
        self.batch_size = batch_size
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.sam2_predictor = sam2_predictor
        
        # Queues for communication between threads
        self.dino_to_sam_queue = Queue(maxsize=5)  # Buffer for 5 batches
        self.result_queue = Queue()
        
        # Flags
        self.stop_flag = False
    
    def dino_worker(self, image_batches, text_queries):
        """Worker thread for DINO processing."""
        try:
            for batch_idx, image_batch in enumerate(image_batches):
                if self.stop_flag:
                    break
                
                # Process with DINO
                inputs = self.dino_processor(
                    text=text_queries[batch_idx:batch_idx+1] * len(image_batch),
                    images=image_batch,
                    return_tensors="pt"
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.dino_model(**inputs)
                
                # Process results to get bounding boxes
                batch_sizes = [(img.shape[0], img.shape[1]) for img in image_batch]
                results = self.dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=batch_sizes
                )
                
                # Extract the first (top) bounding box from each result
                boxes_batch = []
                for result in results:
                    if len(result["boxes"]) > 0:
                        # Convert to x1, y1, x2, y2 format
                        box = result["boxes"][0].tolist()
                        boxes_batch.append(np.array([box]))
                    else:
                        # Default box if no detection
                        h, w = image_batch[0].shape[:2]
                        boxes_batch.append(np.array([[int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]]))
                
                # Put results in queue
                self.dino_to_sam_queue.put((batch_idx, image_batch, boxes_batch))
        except Exception as e:
            print(f"Error in DINO worker: {e}")
        finally:
            # Signal end of processing
            self.dino_to_sam_queue.put(None)
    
    def sam_worker(self):
        """Worker thread for SAM2 processing."""
        try:
            while not self.stop_flag:
                # Get batch from queue
                item = self.dino_to_sam_queue.get()
                if item is None:
                    break
                
                batch_idx, image_batch, boxes_batch = item
                
                # Process with SAM2
                self.sam2_predictor.set_image_batch(image_batch)
                masks_batch, scores_batch, _ = self.sam2_predictor.predict_batch(
                    None, None, box_batch=boxes_batch, multimask_output=False
                )
                
                # Put results in result queue
                self.result_queue.put((batch_idx, masks_batch, scores_batch))
                
                # Signal task completion
                self.dino_to_sam_queue.task_done()
        except Exception as e:
            print(f"Error in SAM worker: {e}")
        finally:
            # Signal end of processing
            self.result_queue.put(None)
    
    def run_parallel_benchmark(self, num_batches=5):
        """Run a benchmark of the parallel pipeline."""
        # Generate test data
        test_image = prepare_fixed_image()
        image_batches = [[test_image] * self.batch_size for _ in range(num_batches)]
        text_queries = [["object"]] * num_batches
        
        # Create and start threads
        dino_thread = Thread(target=self.dino_worker, args=(image_batches, text_queries))
        sam_thread = Thread(target=self.sam_worker)
        
        # Time the entire process
        start_time = time.time()
        dino_thread.start()
        sam_thread.start()
        
        # Collect results (in order)
        results = [None] * num_batches
        received = 0
        
        while received < num_batches and not self.stop_flag:
            item = self.result_queue.get()
            if item is None:
                break
                
            batch_idx, masks_batch, scores_batch = item
            results[batch_idx] = (masks_batch, scores_batch)
            received += 1
            self.result_queue.task_done()
        
        # Wait for threads to finish
        self.stop_flag = True
        dino_thread.join()
        sam_thread.join()
        
        total_time = time.time() - start_time
        throughput = (num_batches * self.batch_size) / total_time
        
        return {
            "total_time": total_time,
            "num_images": num_batches * self.batch_size,
            "throughput": throughput,
            "results": results
        }

def run_parallel_pipeline_benchmark(optimal_batch_size):
    """Run benchmarks for the parallel pipeline."""
    print("\nBenchmarking parallel pipeline with optimal batch size:", optimal_batch_size)
    
    # Load models
    dino_model, dino_processor = load_dino_model()
    sam2_predictor = load_sam2_model()
    print(sam2_predictor)
    print('sam loaded')
    # Create pipeline
    pipeline = ParallelPipeline(
        batch_size=optimal_batch_size,
        dino_model=dino_model,
        dino_processor=dino_processor,
        sam2_predictor=sam2_predictor
    )
    
    # Run warmup
    print("Running warmup...")
    pipeline.run_parallel_benchmark(num_batches=2)
    
    # Run actual benchmark
    print("Running benchmark...")
    num_batches = 5
    results = pipeline.run_parallel_benchmark(num_batches=num_batches)
    
    print(f"\nParallel Pipeline Results:")
    print(f"Total time: {results['total_time']:.4f}s")
    print(f"Images processed: {results['num_images']}")
    print(f"Throughput: {results['throughput']:.2f} images/second")
    
    return results

def main():
    # Test 1: Find maximum batch size


    final_results = benchmark_processing_time(16)
    #benchmark_results = benchmark_processing_time(16)
    
    #pipeline_results = run_parallel_pipeline_benchmark(8)


if __name__ == "__main__":
    main()