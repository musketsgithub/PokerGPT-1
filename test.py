import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def display_image_section(image_path, x_rel, y_rel, width, height):
    # Load image
    img = mpimg.imread(image_path)
    img_height, img_width = img.shape[:2]  # Get image dimensions

    # Convert relative coordinates to absolute pixel positions
    x = int(x_rel * img_width)
    y = int(y_rel * img_height)

    # Extract section
    cropped_img = img[y:y + height, x:x + width]

    # Display image section
    plt.imshow(cropped_img)
    plt.axis('off')  # Hide axes
    plt.show()


# Example usage
image_path = 'dealer_test_cropped/6.png'  # Change this to your image path
x_rel, y_rel = 0.724, 0.504  # Relative position (30% from left, 40% from top)
width, height = 100, 100 # Crop width and height in pixels

display_image_section(image_path, x_rel, y_rel, width, height)
