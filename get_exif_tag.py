import requests
from PIL import Image, ExifTags
from io import BytesIO
import pandas as pd

def get_exif_orientation(image_url):
    try:
        # Download the image (without saving to disk)
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Open the image from memory
        with Image.open(BytesIO(response.content)) as img:
            print(ExifTags.TAGS.keys())
            # Your exact method for finding orientation
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif_data = img._getexif()  # Using _getexif() to match your method
            
            if exif_data is not None and orientation in exif_data:
                orientation_value = exif_data[orientation]
                print(f"Found orientation: {orientation_value} in {image_url}")
                return orientation_value
            else:
                print(f"No orientation data found in {image_url}")
                return None
                
    except Exception as e:
        print(f"Error processing {image_url}: {str(e)}")
        return None

def process_images(csv_path, column_name='image_url', sample_size=100):
    df = pd.read_csv(csv_path)
    image_urls = df[column_name].tolist()
    
    orientation_results = []
    
    for url in image_urls[:sample_size]:  # Process sample by default
        orientation = get_exif_orientation(url)
        orientation_results.append({
            'url': url,
            'orientation': orientation
        })
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(orientation_results)
    results_df.to_csv('orientation_results.csv', index=False)
    print("\nResults saved to orientation_results.csv")
    
    # Print summary statistics
    print("\nOrientation Value Counts:")
    print(results_df['orientation'].value_counts(dropna=False))

if __name__ == "__main__":
    csv_path = '/teamspace/studios/this_studio/big_production_sampling.csv'
    process_images(csv_path)