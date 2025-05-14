import requests
from io import BytesIO
import json
import random
import csv
import argparse
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_url", default='https://prod-k8s.treetracker.org/query/trees?', type=str,
                        help="Query URL for production database.")
    parser.add_argument("--output_file_path", default='big_production_sampling.csv', type=str,
                        help="Path to the output csv.")
    parser.add_argument("--total_count", default=100000, type=int,
                        help='Number of samples to get.')
    parser.add_argument("--query_limit", default=100, type=int,
                        help='Maximum number of samples per API request (cannot exceed 10,000).')
    parser.add_argument("--max_workers", default=5, type=int,
                        help='Number of parallel workers for API requests.')
    parser.add_argument("--max_retries", default=3, type=int,
                        help='Maximum retries for failed API requests.')
    parser.add_argument("--total", default=6573422, type=int,
                        help='total samples in the production database.')
    return parser

def fetch_total_trees(query_url):
    """Fetch the total number of trees using a HEAD request."""
    try:
        response = requests.head(query_url)
        if response.status_code == 200:
            # Extract total count from headers (adjust based on API)
            return int(response.headers.get('X-Total-Count', 0))
        else:
            logger.warning(f"HEAD request failed: {response.status_code}")
            return 0
    except Exception as e:
        logger.error(f"Error fetching total trees: {e}")
        return 0

def fetch_trees_batch(query_url, offset, limit, retries=3):
    """Fetch a batch of trees with retries."""
    for attempt in range(retries):
        try:
            params = {'offset': offset, 'limit': limit}
            response = requests.get(query_url, params=params)
            if response.status_code == 200:
                json_body = json.load(BytesIO(response.content))
                return json_body.get('trees', [])
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}): {response.status_code}")
        except Exception as e:
            logger.warning(f"Error (attempt {attempt + 1}): {e}")
        sleep(2 ** attempt)  # Exponential backoff
    return []

def main():
    opts = get_argparser().parse_args()
    
    # Dynamically fetch total trees if possible
    total_trees = fetch_total_trees(opts.query_url)
    if total_trees <= 0:
        logger.error("Could not fetch total trees. Using default or exiting.")
        total_trees=opts.total
    
    logger.info(f"Total trees in database: {total_trees}")
    
    sampled_urls = set()
    progress_bar = tqdm(total=opts.total_count, desc="Sampling trees")
    
    with ThreadPoolExecutor(max_workers=opts.max_workers) as executor:
        futures = []
        while len(sampled_urls) < opts.total_count:
            # Submit parallel requests
            random_offset = random.randint(0, total_trees - opts.query_limit)
            futures.append(
                executor.submit(
                    fetch_trees_batch,
                    opts.query_url,
                    random_offset,
                    opts.query_limit,
                    opts.max_retries
                )
            )
            
            # Process completed futures
            for future in as_completed(futures):
                tree_data = future.result()
                if tree_data:
                    new_urls = {
                        tree.get('image_url') for tree in tree_data 
                        if tree.get('image_url') and tree.get('image_url') not in sampled_urls
                    }
                    sampled_urls.update(new_urls)
                    progress_bar.update(len(new_urls))
                
                if len(sampled_urls) >= opts.total_count:
                    break
            
            futures = []  # Reset futures list
    
    # Write to CSV
    with open(opts.output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_url"])
        writer.writerows([[url] for url in sampled_urls])
    
    logger.info(f"Sampled {len(sampled_urls)} unique URLs. Saved to {opts.output_file_path}")

if __name__ == "__main__":
    main()