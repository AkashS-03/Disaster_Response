import os
import io
import time
import shutil
import zipfile
import subprocess
from PIL import Image

ZENODO_AIDERV2_TRAIN_URL = "https://zenodo.org/api/records/10891054/files/Train.zip/content"
EXPECTED_ARCHIVE_SIZE = 1195553208  # ~1.14 GB

def prep_vision_data():
    print("=== AIDERv2 Drone Disaster Imagery Pipeline ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    vision_dir = os.path.join(data_dir, "vision")
    
    flood_dir = os.path.join(vision_dir, "flooded")
    clear_dir = os.path.join(vision_dir, "clear")
    zip_path = os.path.join(data_dir, "AIDERv2_Train.zip")
    
    os.makedirs(flood_dir, exist_ok=True)
    os.makedirs(clear_dir, exist_ok=True)
    
    # 1. Download AIDERv2 Train Archive with automatic resume loop
    print(f"\nTarget Archive: {zip_path}")
    print(f"Source URL: {ZENODO_AIDERV2_TRAIN_URL}")
    
    max_retries = 15
    for attempt in range(1, max_retries + 1):
        cur_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        if cur_size >= EXPECTED_ARCHIVE_SIZE:
            print(f"Archive is fully downloaded ({cur_size / (1024*1024):.2f} MB).")
            break
            
        print(f"\n[Attempt {attempt}/{max_retries}] Downloading/Resuming from {cur_size / (1024*1024):.2f} MB...")
        cmd = [
            "curl.exe", "-L", "-C", "-",
            "--retry", "5",
            "--retry-delay", "3",
            "--progress-bar",
            "-o", zip_path,
            ZENODO_AIDERV2_TRAIN_URL
        ]
        
        proc = subprocess.run(cmd)
        
        new_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        if new_size >= EXPECTED_ARCHIVE_SIZE or proc.returncode == 0:
            print(f"\nDownload complete! Final size: {new_size / (1024*1024):.2f} MB.")
            break
        else:
            print(f"Connection dropped. Stored {new_size / (1024*1024):.2f} MB on disk. Reconnecting in 3s...")
            time.sleep(3)

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100_000_000:
        print("Error: Download failed or archive is too small.")
        return

    # 2. Extract and filter Flood and Normal (Clear) drone classes
    print("\nExtracting and validating UAV drone imagery...")
    
    # Clear old legacy files before population
    for folder in [flood_dir, clear_dir]:
        for fname in os.listdir(folder):
            if not fname.startswith("drone_"):
                fpath = os.path.join(folder, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass

    flood_count = 0
    clear_count = 0
    corrupted_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            print(f"Total archive entries to scan: {len(members)}")
            
            for name in members:
                is_flood = ("flood" in name.lower()) and not name.endswith("/")
                is_clear = ("normal" in name.lower()) and not name.endswith("/")
                
                if not (is_flood or is_clear):
                    continue
                    
                ext = os.path.splitext(name)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                    
                try:
                    img_data = zf.read(name)
                    img = Image.open(io.BytesIO(img_data))
                    img.verify()
                    
                    if is_flood:
                        flood_count += 1
                        out_path = os.path.join(flood_dir, f"drone_flood_{flood_count:05d}{ext}")
                        with open(out_path, "wb") as f:
                            f.write(img_data)
                    elif is_clear:
                        clear_count += 1
                        out_path = os.path.join(clear_dir, f"drone_clear_{clear_count:05d}{ext}")
                        with open(out_path, "wb") as f:
                            f.write(img_data)
                            
                except Exception:
                    corrupted_count += 1
                    continue
                    
                if (flood_count + clear_count) % 1000 == 0:
                    print(f"Extracted: {flood_count} flood, {clear_count} clear...")

        print("\n=== Extraction Complete ===")
        print(f"Flooded drone images:      {flood_count}")
        print(f"Clear/Normal drone images:  {clear_count}")
        print(f"Total valid drone images:   {flood_count + clear_count}")
        if corrupted_count > 0:
            print(f"Corrupt files excluded:    {corrupted_count}")

        # 3. Clean up zip archive to reclaim disk space
        if os.path.exists(zip_path):
            print(f"\nCleaning up temporary archive {zip_path}...")
            os.remove(zip_path)
            print("Cleanup finished. Disk space reclaimed.")

        print("\n=== Vision Dataset Preparation Succeeded ===")

    except zipfile.BadZipFile as e:
        print(f"Error: Archive is not yet a complete ZIP file: {e}")

if __name__ == "__main__":
    prep_vision_data()
