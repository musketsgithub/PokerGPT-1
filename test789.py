import cv2
import os


def convert_to_grayscale(file_path):
    # Read the image
    img = cv2.imread(file_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image back to the same location
    cv2.imwrite(file_path, gray_img)
    print(f"Converted {file_path} to grayscale")


# List of image files to process
image_files = [
    # "images/all-in.png",
    # "images/bet.png",
    "images/raise.png",
    "images/check.png",
    "images/fold.png",
    "images/call.png"
]

# Process each file
for file_path in image_files:
    if os.path.exists(file_path):
        convert_to_grayscale(file_path)
    else:
        print(f"File not found: {file_path}")

print("All images processed.")