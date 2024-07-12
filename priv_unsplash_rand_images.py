import requests
import os
import time
import random
from dotenv import load_dotenv

# Load access key from .env file
load_dotenv()
access_key = os.getenv('UNSPLASH_ACCESS_KEY')

queries = ['nature', 'city', 'technology', 'people', 'animals', 'abstract', 'travel', 'food']  # Different queries to add randomness - was getting repeat images with the random query
num_images = 400  # Total number of images to download

base_path = os.getcwd()
output_folder = os.path.join(base_path, "images", "random_pics")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

images_per_request = 30  # rate limit
total_downloaded = 0
downloaded_urls = set() 


while total_downloaded < num_images:
    images_to_download = min(images_per_request, num_images - total_downloaded)
    query = random.choice(queries)  # Choose a random query
    url = f'https://api.unsplash.com/photos/random?count={images_to_download}&query={query}&client_id={access_key}'
    
    response = requests.get(url)
    data = response.json()
    
    for photo in data:
        img_url = photo['urls']['regular']
        
        if img_url in downloaded_urls:
            continue  # Skip if URL is already downloaded
        
        downloaded_urls.add(img_url)
        img_data = requests.get(img_url).content
        img_path = os.path.join(output_folder, f'random_{total_downloaded}.jpg')
        with open(img_path, 'wb') as handler:
            handler.write(img_data)
        print(f'Downloaded {img_path}')
        
        total_downloaded += 1
        if total_downloaded >= num_images:
            break  # break when we have enough images
    
    # Pause before next request
    time.sleep(2)

print(f'Total images downloaded: {total_downloaded}')