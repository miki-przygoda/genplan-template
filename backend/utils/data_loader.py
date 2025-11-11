from datasets import load_dataset
import os
import time
import random

datasets = [ 
    "HamzaWajid1/FloorPlans970Dataset", 
]

def download_datasets():
    """
    Download datasets from Hugging Face and organize them into folders
    based on the dataset name (second part of the repository path).
    Includes rate limiting protection with random delays.
    """
    os.makedirs("backend/data/dataset", exist_ok=True)
    
    for i, dataset_path in enumerate(datasets):
        dataset_name = dataset_path.split("/")[-1]
        dataset_folder = os.path.join("backend", "data", "dataset", dataset_name)
        
        print(f"Downloading {dataset_path}...")
        print(f"Creating folder: {dataset_folder}")
        
        if i > 0:
            delay = random.uniform(2, 5)
            print(f"⏳ Waiting {delay:.1f} seconds to avoid rate limiting...")
            time.sleep(delay)
        
        try:
            os.makedirs(dataset_folder, exist_ok=True)
            
            print("   Loading dataset (this may take a while for large datasets)...")
            dataset = load_dataset(dataset_path, verification_mode="no_checks")
            
            dataset.save_to_disk(dataset_folder)
            
            print(f"✅ Successfully downloaded and saved {dataset_path} to {dataset_folder}")
            print(f"   Dataset info: {dataset}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_path}: {str(e)}")
            
            if "expected" in str(e) and "recorded" in str(e):
                print("   💡 This appears to be a dataset metadata mismatch error.")
                print("   💡 The dataset may have been updated but metadata is outdated.")
                print("   💡 Try downloading this dataset manually or contact the dataset owner.")
            elif "Connection" in str(e) or "timeout" in str(e).lower():
                print("   💡 This appears to be a network connectivity issue.")
                print("   💡 Check your internet connection and try again.")
            else:
                print("   💡 Unknown error - check the dataset URL and try again.")
            
            print("-" * 50)
            
            if i < len(datasets) - 1:
                error_delay = random.uniform(5, 10)
                print(f"⏳ Waiting {error_delay:.1f} seconds before next download...")
                time.sleep(error_delay)

if __name__ == "__main__":
    print("Starting dataset download process...")
    print("=" * 50)
    download_datasets()
    print("=" * 50)
    print("Dataset download process completed!")