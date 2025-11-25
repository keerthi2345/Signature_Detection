import os
from PIL import Image
from tqdm import tqdm

RAW_POS = "data/raw/positive"
RAW_NEG = "data/raw/negative"

PROC_POS = "data/processed/positive"
PROC_NEG = "data/processed/negative"

IMAGE_SIZE = (256, 256)

os.makedirs(PROC_POS, exist_ok=True)
os.makedirs(PROC_NEG, exist_ok=True)

def process_images(src_folder, dst_folder):
    images = os.listdir(src_folder)

    for img_name in tqdm(images, desc=f"Processing {src_folder}", unit="img"):
        src_path = os.path.join(src_folder, img_name)

        # Skip non-images
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            dst_name = os.path.splitext(img_name)[0] + ".png"
            dst_path = os.path.join(dst_folder, dst_name)

            img.save(dst_path, "PNG")

        except Exception as e:
            print("Error processing:", src_path, e)
            continue

print("\nStarting dataset processing...\n")

process_images(RAW_POS, PROC_POS)
process_images(RAW_NEG, PROC_NEG)

print("\nProcessing Complete!")
