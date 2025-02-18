import numpy as np
import cv2
import ctypes
from ctypes import windll, c_void_p, Structure, POINTER, c_int
from ctypes.wintypes import DWORD, LONG, WORD, HWND, HDC, HBITMAP, HGDIOBJ, RECT, BOOL


class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ("biSize", DWORD),
        ("biWidth", LONG),
        ("biHeight", LONG),
        ("biPlanes", WORD),
        ("biBitCount", WORD),
        ("biCompression", DWORD),
        ("biSizeImage", DWORD),
        ("biXPelsPerMeter", LONG),
        ("biYPelsPerMeter", LONG),
        ("biClrUsed", DWORD),
        ("biClrImportant", DWORD)
    ]


class BITMAPINFO(Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", DWORD * 3)
    ]


# Constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


def capture_screen_low_level(region=None):
    """
    Captures screen at lowest available level using Windows GDI

    Args:
        region (tuple): Optional (left, top, right, bottom) coordinates

    Returns:
        numpy.ndarray: Screenshot as a numpy array in BGR format
    """
    # Get handles
    hwnd = windll.user32.GetDesktopWindow()

    # Get dimensions
    if region:
        left, top, right, bottom = region
        width = right - left
        height = bottom - top
    else:
        left, top = 0, 0
        width = windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
        height = windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN

    # Get device contexts
    h_screen_dc = windll.user32.GetDC(hwnd)
    h_memory_dc = windll.gdi32.CreateCompatibleDC(h_screen_dc)

    # Create compatible bitmap
    h_bitmap = windll.gdi32.CreateCompatibleBitmap(h_screen_dc, width, height)

    # Select bitmap into memory DC
    old_obj = windll.gdi32.SelectObject(h_memory_dc, h_bitmap)

    # Copy screen to bitmap
    result = windll.gdi32.BitBlt(
        h_memory_dc, 0, 0, width, height,
        h_screen_dc, left, top, SRCCOPY
    )

    if not result:
        # If BitBlt failed, try alternative method with PrintWindow
        target_hwnd = windll.user32.WindowFromPoint(left + width // 2, top + height // 2)
        if target_hwnd:
            # Use PrintWindow which can capture some protected windows
            rect = RECT()
            windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top

            windll.user32.PrintWindow(target_hwnd, h_memory_dc, 2)  # PW_RENDERFULLCONTENT = 2

    # Create bitmap info structure
    bitmap_info = BITMAPINFO()
    bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bitmap_info.bmiHeader.biWidth = width
    bitmap_info.bmiHeader.biHeight = -height  # Negative for top-down
    bitmap_info.bmiHeader.biPlanes = 1
    bitmap_info.bmiHeader.biBitCount = 24
    bitmap_info.bmiHeader.biCompression = BI_RGB

    # Calculate buffer size (aligned to 4-byte boundary per scan line)
    scan_line_size = (width * 3 + 3) & ~3
    buffer_size = scan_line_size * height

    # Create buffer for pixel data
    buffer = (ctypes.c_char * buffer_size)()

    # Get bitmap bits
    windll.gdi32.GetDIBits(
        h_memory_dc, h_bitmap, 0, height,
        ctypes.byref(buffer), ctypes.byref(bitmap_info), DIB_RGB_COLORS
    )

    # Convert buffer to numpy array
    image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)

    # Clean up
    windll.gdi32.SelectObject(h_memory_dc, old_obj)
    windll.gdi32.DeleteObject(h_bitmap)
    windll.gdi32.DeleteDC(h_memory_dc)
    windll.user32.ReleaseDC(hwnd, h_screen_dc)

    return image_array


def bypass_screenshot_protection(x=0, y=0, width=None, height=None, resize_dimensions=None):
    """
    Captures screen using multiple methods to bypass protection

    Args:
        x, y (int): Top-left coordinates
        width, height (int): Dimensions to capture
        resize_dimensions (tuple): Optional (width, height) to resize output

    Returns:
        numpy.ndarray: Screenshot as numpy array in BGR format
    """
    # Set default width/height if not provided
    if width is None:
        width = windll.user32.GetSystemMetrics(0) - x
    if height is None:
        height = windll.user32.GetSystemMetrics(1) - y

    region = (x, y, x + width, y + height)

    # Try multiple capture methods
    methods = [
        # Method 1: Standard GDI capture
        lambda: capture_screen_low_level(region),

        # Method 2: Try to find window under cursor and capture it
        lambda: capture_window_at_position(region),

        # Method 3: Try to take region from primary display using device driver
        lambda: capture_using_dxgi(region)
    ]

    # Try methods until one works
    screenshot = None
    for method in methods:
        try:
            screenshot = method()
            if screenshot is not None and screenshot.size > 0:
                break
        except Exception as e:
            print(f"Method failed: {e}")
            continue

    # If we got a screenshot and resize is requested
    if screenshot is not None and resize_dimensions:
        screenshot = cv2.resize(screenshot, resize_dimensions, interpolation=cv2.INTER_AREA)

    return screenshot


def capture_window_at_position(region):
    """Try to capture specific window at position"""
    left, top, right, bottom = region
    width = right - left
    height = bottom - top

    # Find window at position
    hwnd = windll.user32.WindowFromPoint(left + width // 2, top + height // 2)
    if not hwnd:
        return None

    # Get window dimensions
    rect = RECT()
    windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

    # Create DC and bitmap
    h_window_dc = windll.user32.GetWindowDC(hwnd)
    h_memory_dc = windll.gdi32.CreateCompatibleDC(h_window_dc)

    w = rect.right - rect.left
    h = rect.bottom - rect.top

    h_bitmap = windll.gdi32.CreateCompatibleBitmap(h_window_dc, w, h)
    old_obj = windll.gdi32.SelectObject(h_memory_dc, h_bitmap)

    # Use PrintWindow which can bypass some protections
    windll.user32.PrintWindow(hwnd, h_memory_dc, 2)  # PW_RENDERFULLCONTENT = 2

    # Create bitmap info
    bitmap_info = BITMAPINFO()
    bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bitmap_info.bmiHeader.biWidth = w
    bitmap_info.bmiHeader.biHeight = -h
    bitmap_info.bmiHeader.biPlanes = 1
    bitmap_info.bmiHeader.biBitCount = 24
    bitmap_info.bmiHeader.biCompression = BI_RGB

    # Calculate buffer size
    scan_line_size = (w * 3 + 3) & ~3
    buffer_size = scan_line_size * h
    buffer = (ctypes.c_char * buffer_size)()

    # Get bitmap bits
    windll.gdi32.GetDIBits(
        h_memory_dc, h_bitmap, 0, h,
        ctypes.byref(buffer), ctypes.byref(bitmap_info), DIB_RGB_COLORS
    )

    # Convert to numpy array
    img_array = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 3)

    # Cleanup
    windll.gdi32.SelectObject(h_memory_dc, old_obj)
    windll.gdi32.DeleteObject(h_bitmap)
    windll.gdi32.DeleteDC(h_memory_dc)
    windll.user32.ReleaseDC(hwnd, h_window_dc)

    # Crop to original region if needed
    if left > rect.left and top > rect.top:
        rel_left = left - rect.left
        rel_top = top - rect.top
        if rel_left < w and rel_top < h:
            crop_right = min(rel_left + width, w)
            crop_bottom = min(rel_top + height, h)
            img_array = img_array[rel_top:crop_bottom, rel_left:crop_right]

    return img_array


def capture_using_dxgi(region):
    """Simpler DXGI capture for Windows 8+ systems"""
    try:
        # Only import if we need it (since it's Windows 8+ only)
        import dxgi
        left, top, right, bottom = region
        width = right - left
        height = bottom - top

        # Create DXGI capture
        capture = dxgi.DXGICapture()
        img = capture.grab()

        # Crop if needed
        if left > 0 or top > 0:
            img = img[top:bottom, left:right]

        return img
    except ImportError:
        # DXGI import failed, return None to try next method
        return None
    except Exception as e:
        print(f"DXGI capture failed: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Try with specific region
    x, y = 100, 100
    width, height = 800, 600

    # Take screenshot
    screenshot = bypass_screenshot_protection(x, y, width, height)

    if screenshot is not None:
        # Save the screenshot
        cv2.imwrite("poker_screenshot.png", screenshot)
        print("Screenshot saved successfully!")
    else:
        print("All screenshot methods failed!")