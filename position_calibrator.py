import tkinter as tk
from tkinter import filedialog, Canvas, Scrollbar
from PIL import Image, ImageTk
import pywinctl as gw
import platform


class PokerScreenshotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Window Screenshot Viewer")

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = Canvas(self.canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y = Scrollbar(self.canvas_frame, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x = Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.config(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.image_tk = None
        self.image_pil = None
        self.cropped_image = None  # Cached cropped image
        self.zoom_factor = 1.0
        self.start_x = None
        self.start_y = None
        self.rect = None

        self.upload_button = tk.Button(root, text="Upload Screenshot", command=self.upload_screenshot)
        self.upload_button.pack()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def upload_screenshot(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.image_pil = Image.open(filepath)
            self.crop_poker_window()
            self.display_image()

    def crop_poker_window(self):
        """Crop the uploaded screenshot to the poker window dimensions (if found)."""
        if not self.image_pil:
            return

        windows = [w for w in gw.getAllWindows() if "nlhp" in w.title.lower()]
        if windows:
            window = windows[0]
            left, top, width, height = window.left, window.top, window.width, window.height

            if platform.system() == "Darwin":
                left, top, width, height = left * 2, top * 2, width * 2, height * 2

            # Cache cropped image so we don’t reprocess it every time
            self.cropped_image = self.image_pil.crop((left, top, left + width, top + height))
        else:
            self.cropped_image = self.image_pil  # Use the full image if no window found

    def display_image(self):
        """Efficiently redraw image without lagging."""
        if not self.cropped_image:
            return

        # Resize only the cached image for zoom
        new_size = (int(self.cropped_image.width * self.zoom_factor),
                    int(self.cropped_image.height * self.zoom_factor))
        resized_image = self.cropped_image.resize(new_size, Image.LANCZOS)

        # Convert to Tkinter format
        self.image_tk = ImageTk.PhotoImage(resized_image)

        # Clear only the previous image, not all canvas objects
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk, tags="image")

        # Update scroll region to match new image size
        self.canvas.config(scrollregion=(0, 0, resized_image.width, resized_image.height))

    def on_click(self, event):
        """Start rectangle selection or log click coordinates."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        """Update selection rectangle while dragging."""
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, current_x, current_y)

    def on_release(self, event):
        """Handle mouse release for selecting areas or logging coordinates."""
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)

        if abs(current_x - self.start_x) < 5 and abs(current_y - self.start_y) < 5:
            x = self.start_x / (self.cropped_image.width * self.zoom_factor)
            y = self.start_y / (self.cropped_image.height * self.zoom_factor)
            print(f"Clicked at relative coordinates: x={x:.3f}, y={y:.3f}")
        else:
            width = abs(current_x - self.start_x)
            height = abs(current_y - self.start_y)
            print(f"Selected area: width={width:.0f} pixels, height={height:.0f} pixels")

        self.start_x = None
        self.start_y = None
        self.rect = None

    def on_mousewheel(self, event):
        """Efficiently zoom in/out without lag."""
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1

        # Prevent excessive zooming out
        self.zoom_factor = max(0.2, min(self.zoom_factor, 5))

        # Only resize, don't recrop
        self.display_image()


root = tk.Tk()
app = PokerScreenshotViewer(root)
root.mainloop()
