import cv2
import os


def crop_images(directory, crop_params, new_directory):
    """
    Crops six image files in the given directory based on individual crop_params and saves them in a new directory.

    :param directory: Path to the directory containing the images.
    :param crop_params: Dictionary mapping filenames to (x, y, width, height) for cropping.
    :param new_directory: Path to the directory where cropped images will be saved.
    """
    if not os.path.exists(new_directory):  # Ensure new directory exists
        os.makedirs(new_directory)

    files = [f for f in os.listdir(directory)]

    for file in files[:6]:  # Process only the first six images
        img_path = os.path.join(directory, file)  # Read from original directory
        print(f"Processing: {img_path}")  # Debugging statement

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {file}. Check if the file exists and is a valid image.")
            continue

        if file not in crop_params:
            print(f"No crop parameters provided for {file}")
            continue

        x, y, w, h = crop_params[file]
        cropped_img = img[y:y + h, x:x + w]  # Crop the image

        new_img_path = os.path.join(new_directory, file)  # Save to new directory
        cv2.imwrite(new_img_path, cropped_img)
        print(f"Cropped and saved: {new_img_path}")


# Example usage
directory = "dealer_test"  # Change this to your actual directory path
new_directory = "dealer_test_cropped"
crop_params = {
    "1.png": (916, 592, 948, 710),
    "2.png": (916, 592, 948, 710),
    "3.png": (916, 592, 948, 710),
    "4.png": (916, 592, 948, 710),
    "5.png": (916, 592, 948, 710),
    "6.png": (916, 592, 948, 710)
}  # Example crop params per image

crop_images(directory, crop_params, new_directory)
