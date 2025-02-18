import os
import openai
import pywinctl as gw
from colorama import init
import tkinter as tk
import threading

from game_state import GameState
from gui import GUI
from hero_action import HeroAction
from poker_assistant import PokerAssistant
from audio_player import AudioPlayer
from read_poker_table import ReadPokerTable
from hero_hand_range import PokerHandRangeDetector
from hero_info import HeroInfo


class PokerWindowAnalyzer(threading.Thread):
    def __init__(self, window, game_window, openai_client, hero_player_number=1):
        super().__init__()
        self.window = window
        self.game_window = game_window
        self.openai_client = openai_client
        self.hero_player_number = hero_player_number
        self.running = True

        # Initialize components
        self.audio_player = AudioPlayer(openai_client)
        self.hero_action = HeroAction(window)
        self.hero_info = HeroInfo()
        self.hero_hand_range = PokerHandRangeDetector()
        self.game_state = GameState(self.hero_action, self.audio_player)

        # Set up poker assistant
        self.poker_assistant = PokerAssistant(
            openai_client,
            self.hero_info,
            self.game_state,
            self.hero_action,
            self.audio_player
        )

        # Initialize GUI
        self.gui = GUI(self.game_state, self.poker_assistant, parent_window=self.game_window)

        # Initialize table reader
        self.read_poker_table = ReadPokerTable(
            window,
            self.hero_info,
            self.hero_hand_range,
            self.hero_action,
            self.poker_assistant,
            self.game_state
        )

        # Set up initial game state
        self.game_state.update_player(hero_player_number, hero=True)
        self.game_state.hero_player_number = hero_player_number
        self.game_state.extract_blinds_from_title()

    def run(self):
        """Main thread execution"""
        try:
            # Start continuous detection
            self.read_poker_table.start_continuous_detection()

            # Keep thread running
            while self.running:
                if not self.window.visible:
                    self.stop()
                    break
                threading.Event().wait(0.1)

        except Exception as e:
            print(f"Error in analyzer thread: {e}")

    def stop(self):
        """Stop the analyzer thread"""
        self.running = False
        self.read_poker_table.shutdown_flag.set()


class MultiWindowManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main root window
        self.analyzers = []

    def create_game_instance(self, window, openai_client, hero_player_number=1):
        # Create a new Toplevel window for this instance
        game_window = tk.Toplevel(self.root)

        # Set window title to match poker window
        game_window.title(window.title)

        # Calculate position based on number of existing windows
        window_count = len(self.analyzers)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(screen_width // 2, 1200)
        window_height = min(screen_height // 2, 800)

        row = window_count // 2
        col = window_count % 2
        x_position = col * (window_width + 20)
        y_position = row * (window_height + 20)

        game_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Create and start analyzer thread
        analyzer = PokerWindowAnalyzer(window, game_window, openai_client, hero_player_number)
        analyzer.daemon = True  # Make thread daemon so it closes with main program
        analyzer.start()

        self.analyzers.append(analyzer)

    def run(self):
        """Start the main event loop"""
        try:
            self.root.mainloop()
        finally:
            # Clean up analyzers when program closes
            for analyzer in self.analyzers:
                analyzer.stop()


def analyze_hand_with_openai(openai_client, hand_history):
    """Analyze a hand history with OpenAI."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a poker analysis expert. Analyze the given hand history and provide strategic insights."},
                {"role": "user", "content": f"Analyze this poker hand:\n{hand_history}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing hand: {str(e)}"


def locate_poker_windows():
    """Locate all poker client windows."""
    windows = [w for w in gw.getAllWindows() if "no limit" in w.title.lower()]
    poker_windows = []

    for window in windows:
        if "USD" in window.title or "Money" in window.title:
            print(f"Poker client window found: {window.title}")
            default_width = 963
            default_height = 692
            window.resizeTo(default_width, default_height)
            poker_windows.append(window)

    if not poker_windows:
        print("No poker client windows found.")
    else:
        print(f"Found {len(poker_windows)} poker windows")

    return poker_windows


def main():
    # Initialize OpenAI client
    api_key = "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"
    openai_client = openai.OpenAI(api_key=api_key)
    init(autoreset=True)

    # Find all poker windows
    poker_windows = locate_poker_windows()

    if not poker_windows:
        print("No poker windows found. Exiting...")
        return

    # Create window manager
    manager = MultiWindowManager()

    # Create game instances for each window
    for i, window in enumerate(poker_windows):
        print(f"Initializing game instance {i + 1} for window: {window.title}")
        manager.create_game_instance(window, openai_client, hero_player_number=1)

    # Start the main event loop
    manager.run()


if __name__ == "__main__":
    main()