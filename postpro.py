import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
MODEL_PATH = "d4.pt"             
INPUT_DIR = "raw_images"           # Folder containing drone photos
OUTPUT_DIR = "processed_results"   # Folder where labeled photos will go
CONF_THRESHOLD = 0.4              
IOU_THRESHOLD = 0.5                
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SETUP ---
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
print(f"🔄 Loading model from {MODEL_PATH} to {DEVICE}...")
model = YOLO(MODEL_PATH)

# Warmup: Run one fake inference to wake up the GPU
# (This prevents the first image from being slow)
model.predict(source="https://ultralytics.com/images/bus.jpg", device=DEVICE, verbose=False)
print("✅ Model loaded and GPU warmed up!")

# --- PROCESSING LOOP ---
# Get list of image files
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

print(f"🚀 Starting processing on {len(image_files)} images...")

# Use tqdm for a nice progress bar
for img_file in tqdm(image_files, desc="Processing"):
    input_path = os.path.join(INPUT_DIR, img_file)
    output_path = os.path.join(OUTPUT_DIR, img_file)

    try:
        # 1. Run Inference
        # save=True automatically saves it, but we want control, so we plot manually
        results = model.predict(
            source=input_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,  # Keep console clean
            imgsz=640
        )

        # 2. Visualize & Save
        for result in results:
            # Plot the boxes on the image (numpy array)
            # labels=True (names), conf=True (scores)
            annotated_frame = result.plot(line_width=2, font_size=1.0)
            
            # Save the image using OpenCV
            cv2.imwrite(output_path, annotated_frame)
            
            # Optional: Save text file with detections (if you need data for reports)
            # result.save_txt(output_path.replace('.jpg', '.txt'))

    except Exception as e:
        print(f"⚠️ Error processing {img_file}: {e}")

print(f"\n✅ All done! Check the '{OUTPUT_DIR}' folder.")