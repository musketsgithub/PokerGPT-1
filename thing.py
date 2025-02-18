import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import simpledialog

# Use TkAgg backend
matplotlib.use('TkAgg')


def rename_images(directory):
    # Create main Tk window that will persist throughout the program
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate position for centered window
    position_x = int(screen_width / 2 - 400)  # 400 is half the window width
    position_y = int(screen_height / 2 - 300)  # 300 is half the window height

    image_files = [f for f in os.listdir(directory) if
                   os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(
                       ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort()

    index = 0
    while index < len(image_files) and index >= 0:
        filepath = os.path.join(directory, image_files[index])
        try:
            # Create a new Toplevel window for each image
            img_window = tk.Toplevel(root)
            img_window.title(image_files[index])

            # Set the window position to be consistent
            img_window.geometry(f"800x600+{position_x}+{position_y}")

            # Load and display the image
            img = mpimg.imread(filepath)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.set_title(image_files[index])
            ax.axis('off')

            # Embed the matplotlib figure in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=img_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update the window to ensure it's displayed
            img_window.update()

            # Position the dialog window near the image window
            root.geometry(f"+{position_x + 200}+{position_y + 200}")

            # Use Tkinter's dialog for input
            new_name = simpledialog.askstring("Rename Image",
                                              f"Enter new name for {image_files[index]}\n(leave empty to skip, type 'back' to go back)",
                                              parent=img_window)

            # Close the window after getting input
            plt.close(fig)
            img_window.destroy()

            if new_name is None or new_name == "":
                index += 1
                continue
            elif new_name.lower() == "back":
                index -= 1
                continue
            else:
                new_filepath = os.path.join(directory, new_name)
                extension = os.path.splitext(image_files[index])[1]
                if not new_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    new_filepath += extension
                os.rename(filepath, new_filepath)
                print(f"Renamed {image_files[index]} to {os.path.basename(new_filepath)}")
                image_files[index] = os.path.basename(new_filepath)
                index += 1

        except Exception as e:
            print(f"Error processing {image_files[index]}: {e}")
            index += 1

    # Destroy the main window at the end
    root.destroy()


# Example:
directory_path = "card_images"  # Replace with your directory
rename_images(directory_path)



















































































































































































































































































































































































































































































































































































