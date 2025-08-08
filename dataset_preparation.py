import os
import pandas as pd
import requests
from tqdm import tqdm

# Load and clean CSV
df = pd.read_csv("all_image_urls.csv", header=None, names=["url"])
df = df.dropna().reset_index(drop=True)

# Define class categories and ranges
category_ranges = {
    "plastic": (0, 49),
    "organic": (50, 99),
    "recyclable": (100, 149),
    "hazardous": (150, 199)
}

# Create base directory for dataset
base_dir = "waste_dataset"
os.makedirs(base_dir, exist_ok=True)

# Loop through categories and download images
for category, (start, end) in category_ranges.items():
    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    for i in tqdm(range(start, min(end + 1, len(df))), desc=f"Downloading {category}"):
        try:
            url = df.loc[i, "url"]
            filename = f"{category}_{i}.jpg"
            save_path = os.path.join(category_dir, filename)

            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Skipped {url}: Status {r.status_code}")
        except Exception as e:
            print(f"Error at index {i}: {e}")
