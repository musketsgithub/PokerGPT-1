import os
import openai
import pywinctl as gw
from colorama import init
import tkinter as tk
import threading
import asyncio
from tkinter import ttk
import time
import queue
import sys
import platform
import pygame

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Set macOS-specific environment variables
if platform.system() == 'Darwin':  # macOS
    os.environ['SDL_VIDEODRIVER'] = 'cocoa'
    os.environ['SDL_WINDOWID'] = '0'
    os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    os.environ['SDL_VIDEO_WINDOW_TITLE'] = 'PokerGPT'
    os.environ['SDL_VIDEO_WINDOW_SHOWN'] = '0'
    os.environ['SDL_VIDEO_HIDPI'] = '0'
    os.environ['SDL_VIDEO_WINDOW_MINIMIZED'] = '1'

from read_poker_table import ReadPokerTable
from hero_info import HeroInfo
from hero_hand_range import PokerHandRangeDetector
from hero_action import HeroAction
from poker_assistant import PokerAssistant
from game_state import GameState
from audio_player import AudioPlayer

class AsyncThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.running = False
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()

    def create_task(self, coro):
        if not self.loop.is_running():
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

class PokerGUI:
    def __init__(self):
        # Initialize pygame after Tkinter window is created
        pygame.init()

        self.root = tk.Tk()
        self.root.title("PokerGPT")
        self.root.geometry("800x600")

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create text widget for logging
        self.log_text = tk.Text(self.main_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Create control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.dstart_button = ttk.Button(self.button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(self.button_frame, text="Clear Log", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Find poker window first
        self.poker_window = self.find_poker_window()
        if not self.poker_window:
            self.log_message("No poker window found. Please open a poker window first.")
            self.start_button.state(['disabled'])
            return

        # Initialize components in a thread-safe manner
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components in a thread-safe manner."""
        try:
            # Initialize OpenAI client first
            self.openai_client = openai.OpenAI()

            # Initialize audio player with openai_client
            self.audio_player = AudioPlayer(self.openai_client)

            # Initialize basic components
            self.hero_info = HeroInfo()
            self.hero_hand_range = PokerHandRangeDetector()
            self.hero_action = HeroAction(self.poker_window)

            # Initialize game state with required components
            self.game_state = GameState(
                self.hero_action,
                self.audio_player,
                self.poker_window
            )

            # Initialize poker assistant with all required components
            self.poker_assistant = PokerAssistant(
                self.openai_client,
                self.hero_info,
                self.game_state,
                self.hero_action,
                self.audio_player
            )

            # Initialize read poker table
            self.read_poker_table = ReadPokerTable(
                self.poker_window,
                self.hero_info,
                self.hero_hand_range,
                self.hero_action,
                self.poker_assistant,
                self.game_state
            )

            # Initialize async thread
            self.async_thread = AsyncThread()
            self.async_thread.start()

            # Bind window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Flag to track if detection is running
            self.detection_running = False

        except Exception as e:
            self.log_message(f"Error initializing components: {str(e)}")
            raise

    def find_poker_window(self):
        """Find the poker window."""
        windows = [w for w in gw.getAllWindows() if "nlhp" in w.title.lower()]
        if windows:
            self.log_message(f"Found poker window: {windows[0].title}")
            return windows[0]
        return None

    def start_detection(self):
        """Start the poker table detection."""
        if not self.detection_running:
            # Check if poker window is still available
            if not self.poker_window or not self.poker_window.visible:
                self.poker_window = self.find_poker_window()
                if not self.poker_window:
                    self.log_message("No poker window found. Please open a poker window first.")
                    return

            self.detection_running = True
            self.start_button.state(['disabled'])
            self.stop_button.state(['!disabled'])
            self.log_message("Starting poker table detection...")

            # Create and run the async task
            future = self.async_thread.create_task(self.read_poker_table.start_detection())
            if future:
                future.add_done_callback(self.handle_task_completion)

    def stop_detection(self):
        """Stop the poker table detection."""
        if self.detection_running:
            self.detection_running = False
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.log_message("Stopping poker table detection...")

            # Create and run the shutdown task
            future = self.async_thread.create_task(self.read_poker_table.initiate_shutdown())
            if future:
                future.add_done_callback(self.handle_task_completion)

    def handle_task_completion(self, future):
        """Handle completion of async tasks."""
        try:
            future.result()
        except Exception as e:
            self.log_message(f"Task error: {str(e)}")

    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.delete(1.0, tk.END)

    def log_message(self, message):
        """Add a message to the log with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def on_closing(self):
        """Handle window closing."""
        if self.detection_running:
            self.stop_detection()
        self.async_thread.stop()
        pygame.quit()
        self.root.quit()

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        finally:
            pygame.quit()

def main():
    # Set up the event loop
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Create and run the GUI
    gui = PokerGUI()
    gui.run()

if __name__ == "__main__":
    main()