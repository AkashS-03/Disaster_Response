import os
from icrawler.builtin import GoogleImageCrawler

def prep_vision_data():
    print("--- Fetching Real Labeled Flood Imagery ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vision_dir = os.path.join(base_dir, "data", "vision")
    
    flood_dir = os.path.join(vision_dir, "flooded")
    clear_dir = os.path.join(vision_dir, "clear")
    os.makedirs(flood_dir, exist_ok=True)
    os.makedirs(clear_dir, exist_ok=True)
    
    print("Fetching 'flooded' class images...")
    flood_crawler = GoogleImageCrawler(storage={'root_dir': flood_dir}, downloader_threads=4)
    flood_crawler.crawl(keyword='flood urban street water damage', max_num=30)
    
    print("Fetching 'clear' class images...")
    clear_crawler = GoogleImageCrawler(storage={'root_dir': clear_dir}, downloader_threads=4)
    clear_crawler.crawl(keyword='clear dry sunny city street', max_num=30)
    
    # Validation
    num_flooded = len(os.listdir(flood_dir))
    num_clear = len(os.listdir(clear_dir))
    
    print(f"Vision Data Prep Complete. Dataset size: {num_flooded} flooded, {num_clear} clear.")

if __name__ == "__main__":
    prep_vision_data()
