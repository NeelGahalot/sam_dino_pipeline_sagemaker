import os
import time
import numpy as np
import torch
import gc
from PIL import Image
from queue import Queue
from threading import Thread, Lock
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
TEST_IMAGE_SIZE = (512, 512)
SAM2_CHECKPOINT = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
NUM_WORKERS = 8
NUM_RUNS = 5
NUM_WARMUP = 2
BATCH_SIZES = [1, 2]

def prepare_fixed_image(target_size=TEST_IMAGE_SIZE):
    """Create a fixed test image."""
    try:
        image = Image.open(f'{HOME}/sam2/notebooks/images/truck.jpg')
        image = image.convert("RGB").resize(target_size)
    except:
        image = Image.new("RGB", target_size, color=(128, 128, 128))
    return np.array(image)

def generate_test_box(image_shape):
    """Generate a sample bounding box."""
    h, w = image_shape[:2]
    return np.array([[int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]])

def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class WorkerBenchmark:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.test_image = prepare_fixed_image()
        self.test_box = generate_test_box(self.test_image.shape)
        self.text_query = ["object"]
        self.lock = Lock()
        
    def worker_init(self):
        """Initialize models for each worker (thread-safe)"""
        with self.lock:
            sam_predictor = load_sam2_model()
            dino_model, dino_processor = load_dino_model()
        return sam_predictor, dino_model, dino_processor

    def sam_worker(self, input_queue, output_queue):
        """SAM worker thread function"""
        sam_predictor, _, _ = self.worker_init()
        while True:
            item = input_queue.get()
            if item is None:  # Termination signal
                break
                
            idx, image = item
            try:
                sam_predictor.set_image(image)
                masks, scores, _ = sam_predictor.predict(
                    None, None, box=self.test_box, multimask_output=False
                )
                output_queue.put((idx, True))
            except Exception as e:
                print(f"SAM worker error: {e}")
                output_queue.put((idx, False))
            input_queue.task_done()

    def dino_worker(self, input_queue, output_queue):
        """DINO worker thread function"""
        _, dino_model, dino_processor = self.worker_init()
        while True:
            item = input_queue.get()
            if item is None:  # Termination signal
                break
                
            idx, image = item
            try:
                inputs = dino_processor(
                    text=[self.text_query],
                    images=[image],
                    return_tensors="pt"
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                output_queue.put((idx, True))
            except Exception as e:
                print(f"DINO worker error: {e}")
                output_queue.put((idx, False))
            input_queue.task_done()

    def run_benchmark(self, batch_size, mode='sam'):
        """Run benchmark with parallel workers"""
        # Create queues
        input_queue = Queue()
        output_queue = Queue()
        
        # Create and start workers
        workers = []
        worker_func = self.sam_worker if mode == 'sam' else self.dino_worker
        
        for _ in range(self.num_workers):
            t = Thread(target=worker_func, args=(input_queue, output_queue))
            t.start()
            workers.append(t)
        
        # Prepare tasks (each worker will process batch_size/num_workers images)
        tasks = [(i, self.test_image) for i in range(batch_size)]
        
        # Warmup
        for _ in range(NUM_WARMUP):
            for task in tasks:
                input_queue.put(task)
            for _ in tasks:
                output_queue.get()
        
        # Actual benchmark
        times = []
        for _ in range(NUM_RUNS):
            clear_gpu_memory()
            start_time = time.time()
            
            # Submit tasks
            for task in tasks:
                input_queue.put(task)
            
            # Wait for completion
            for _ in tasks:
                output_queue.get()
            
            times.append(time.time() - start_time)
        
        # Cleanup
        for _ in range(self.num_workers):
            input_queue.put(None)
        for t in workers:
            t.join()
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        return avg_time, throughput

    def benchmark_all(self):
        """Run all benchmarks"""
        print(f"\nBenchmarking with {self.num_workers} workers")
        
        # SAM only
        print("\nSAM Only:")
        for bs in BATCH_SIZES:
            time, throughput = self.run_benchmark(bs, mode='sam')
            print(f"Batch {bs}: {time:.4f}s ({throughput:.2f} img/s)")
        
        # DINO only
        print("\nDINO Only:")
        for bs in BATCH_SIZES:
            time, throughput = self.run_benchmark(bs, mode='dino')
            print(f"Batch {bs}: {time:.4f}s ({throughput:.2f} img/s)")
        
        # Combined (serial execution of DINO then SAM)
        print("\nCombined Pipeline (Serial):")
        for bs in BATCH_SIZES:
            # Time DINO phase
            dino_time, _ = self.run_benchmark(bs, mode='dino')
            # Time SAM phase
            sam_time, _ = self.run_benchmark(bs, mode='sam')
            total_time = dino_time + sam_time
            throughput = bs / total_time
            print(f"Batch {bs}: {total_time:.4f}s ({throughput:.2f} img/s)")

def load_sam2_model():
    """Load SAM2 model"""
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(sam2_model)

def load_dino_model():
    """Load DINO model"""
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    return model, processor

if __name__ == "__main__":
    benchmark = WorkerBenchmark(num_workers=NUM_WORKERS)
    benchmark.benchmark_all()