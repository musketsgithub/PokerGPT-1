import os
import cv2

# Define directory and target size
image_dir = "./images"
target_size = (10, 7)

# List of card file names
card_files = ["Clover.png","Hearts.png","Spades.png","Diamonds.png"]

# Ensure output directory exists
output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for file_name in card_files:
    file_path = os.path.join(image_dir, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Read image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error loading image: {file_path}")
        continue

    # Resize image
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Save resized image
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, img_resized)
    print(f"Resized and saved: {output_path}")

print("Processing complete.")