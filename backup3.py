import mss
import re
import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime

import threading
from threading import Lock

import time

import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt

import time
from skimage.metrics import structural_similarity as ssim

import cProfile
import threading
import time


matplotlib.use('TkAgg')

import os
os.environ['SDL_VIDEODRIVER'] = 'x11'  # Add this at the very top of your file

import pytesseract

class ReadPokerTable:

    def __init__(self, poker_window, hero_info, hero_hand_range, hero_action, poker_assistant, game_state):

        self.game_state_lock = Lock()  # Initialize the lock for game_state

        self.hero_info = hero_info

        self.hero_hand_range = hero_hand_range

        self.hero_action = hero_action

        self.game_state = game_state

        self.poker_assistant = poker_assistant

        self.save_screenshots = False

        self.tesseract_cmd = r'C:\Users\Admin\Desktop\PokerGPT\tesseract\tesseract.exe'

        self.cards_on_table = False

        self.previous_hashes = {}  # Dictionary to store previous hashes for each player

        self.photo = None

        self.last_active_player = 1  # Default to player 1 or any other suitable default

        self.last_action_player = 0  # For detecting player actions and stack sizes only once

        self.hero_buttons_active = {}  # Detected active hero buttons

        self.action_processed = False  # Detect heros action only once when action buttons become active

        self.last_detected_cards = []  # Add a local variable to keep track of the last detected cards

        self.window = poker_window

        self.poker_window_width = self.window.width

        self.poker_window_height = self.window.height

        self.poker_window_left = self.window.left

        self.poker_window_top = self.window.top

        self.window_activation_error_reported = False

        # Load the images
        self.dealer_button_image = cv2.imread('images/dealer_button.png', cv2.IMREAD_GRAYSCALE)

        self.card_icon_templates = {
            '♣': cv2.imread('images/Clover.png', cv2.IMREAD_GRAYSCALE),  # Clubs
            '♦': cv2.imread('images/Diamonds.png', cv2.IMREAD_GRAYSCALE),  # Diamonds
            '♥': cv2.imread('images/Hearts.png', cv2.IMREAD_GRAYSCALE),  # Hearts
            '♠': cv2.imread('images/Spades.png', cv2.IMREAD_GRAYSCALE)  # Spades
        }

        self.card_number_templates = {
            '2': cv2.imread('images/2.png', cv2.IMREAD_GRAYSCALE),
            '3': cv2.imread('images/3.png', cv2.IMREAD_GRAYSCALE),
            '4': cv2.imread('images/4.png', cv2.IMREAD_GRAYSCALE),
            '5': cv2.imread('images/5.png', cv2.IMREAD_GRAYSCALE),
            '6': cv2.imread('images/6.png', cv2.IMREAD_GRAYSCALE),
            '7': cv2.imread('images/7.png', cv2.IMREAD_GRAYSCALE),
            '8': cv2.imread('images/8.png', cv2.IMREAD_GRAYSCALE),
            '9': cv2.imread('images/9.png', cv2.IMREAD_GRAYSCALE),
            '10': cv2.imread('images/10.png', cv2.IMREAD_GRAYSCALE),
            'A': cv2.imread('images/A.png', cv2.IMREAD_GRAYSCALE),
            'J': cv2.imread('images/J.png', cv2.IMREAD_GRAYSCALE),
            'Q': cv2.imread('images/Q.png', cv2.IMREAD_GRAYSCALE),
            'K': cv2.imread('images/K.png', cv2.IMREAD_GRAYSCALE)
        }

        self.progress_bar_template = cv2.imread('images/halfwaybar.png', cv2.IMREAD_GRAYSCALE)

        self.shutdown_flag = threading.Event()

        self.threads = []

        self.scaling_factor = 2

        self.screenshot_lock = Lock()

        self.last_action_time = {}

        self.update_intervals = {  # Store update intervals for each function
            "player_actions": 100,
            "table_cards": 700,
            "player_cards": 1000,
            "player_turn": 100,
            "total_pot_size": 600,
            "hero_action_buttons": 300,
            "dealer_button": 300
        }

        self.image_processing_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.shutdown_flag = threading.Event()

        self.start_image_processing_thread()
        self.start_gui_update_thread()

    def activate_window(self):
        """Activate the poker client window."""

        if self.window:
            try:
                self.window.activate()
                self.window_activation_error_reported = False  # Reset the flag if activation is successful
            except Exception as e:
                if not self.window_activation_error_reported:
                    print(f"Error activating window: {e}")
                    self.window_activation_error_reported = True
        else:
            if not self.window_activation_error_reported:
                print("Window not located or cannot be activated.")
                self.window_activation_error_reported = True

    def create_overlay(self):
        """Create an overlay window with an image on the poker client."""
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Create the overlay as a Toplevel window and assign it to self.overlay
        self.overlay = tk.Toplevel(root)
        self.overlay.title("Poker Overlay")
        self.overlay.geometry(f'{self.window.width}x{self.window.height}+{self.window.left}+{self.window.top}')

        # Load the image from the 'images/' directory
        image_path = 'images/PokerTable.png'
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image, master=self.overlay)

        label = tk.Label(self.overlay, image=self.photo)
        label.pack(fill=tk.BOTH, expand=True)

        self.overlay.overrideredirect(True)
        self.overlay.wm_attributes("-topmost", True)

        # Schedule an update method if needed, for example, every 1000ms
        self.overlay.after(1000, self.overlay_update_method)

        root.mainloop()

    def overlay_update_method(self):
        # Update anything related to the overlay if needed
        pass

    def is_pixel_white(self, pixel, min_white=230, max_white=255):
        """
        Check if a pixel is within the white range.
        """
        r, g, b = pixel
        return all(min_white <= value <= max_white for value in (r, g, b))

    def capture_screen_area(self, relative_x, relative_y, width, height, resize_dimensions=None, filename=None):
        """
        Capture a screen area based on relative coordinates for position and fixed pixel values for size.
        """
        # Log the current size of the window
        # print(f"Current window size: {self.window.width}x{self.window.height}")

        # Calculate absolute position based on cached relative coordinates
        abs_x = int(self.poker_window_left + self.poker_window_width * relative_x)
        abs_y = int(self.poker_window_top + self.poker_window_height * relative_y)

        with mss.mss() as sct:
            monitor = {"top": abs_y, "left": abs_x, "width": width, "height": height}
            screenshot = sct.grab(monitor)

            screenshot = np.array(screenshot)

            if resize_dimensions:
                screenshot = cv2.resize(screenshot, resize_dimensions, interpolation=cv2.INTER_AREA)

            # if self.save_screenshots or filename:
            #     filepath = f'Screenshots/{filename if filename else datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.png'
            #     mss.tools.to_png(screenshot.rgb, screenshot.size, output=filepath)

        # end_time = time.time()  # End timing
        # elapsed_time = end_time - start_time

        # print(f"Screenshot> {elapsed_time:.2f} seconds.")  # Log the time taken

        return screenshot

    def contains_white(self, image):
        """
        Check if the image contains any pixels within the specified white color range.
        """
        self.white_color_lower = np.array([210, 210, 210])
        self.white_color_upper = np.array([218, 218, 218])

        # Create a mask for the white color range
        white_mask = cv2.inRange(image, self.white_color_lower, self.white_color_upper)

        # Check if there are any white pixels in the image
        return np.any(white_mask)

    def contains_blue(self, image):
        """Check if the image contains any pixels within the specified blue color ranges."""

        # Define the lower and upper bounds of the blue colors
        blue_color_1_lower = np.array([100, 167, 195])
        blue_color_1_upper = np.array([110, 185, 216])

        # Create a mask for the blue color range
        blue_mask = cv2.inRange(image, blue_color_1_lower, blue_color_1_upper)

        # Check if there are any blue pixels in the image
        if np.any(blue_mask):
            return True
        else:
            return False

    def detect_hero_buttons(self):
        """
        Check the three buttons for the presence of the specified white color and detect text.
        Updates self.hero_buttons_active with the current state of hero buttons.
        """

        button_width = 120
        button_height = 50

        button_positions = [
            (0.516, 0.907),  # Button1 = contains 'Fold'
            (0.679, 0.907),  # Button2 = contains 'Check' or 'Call'
            (0.842, 0.907),  # Button3 = contains 'Raise' or 'Call' or 'Bet'
        ]

        any_button_active = False
        button_offset = 0.06

        for i, (x, y) in enumerate(button_positions, start=1):
            screenshot = self.capture_screen_area(x, y + button_offset, 1, 1)

            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

            if self.contains_white(screenshot):
                button_text = self.detect_text(x, y, button_width, button_height)

                if button_text:
                    cleaned_button_text = button_text.replace('\n', ' ')

                    print(f"Button {i} Text: {cleaned_button_text}")

                    if "Raise" in cleaned_button_text:
                        self.hero_buttons_active[i] = {"action": "Raise", "pos": (x, y)}
                        any_button_active = True
                    elif "Bet" in cleaned_button_text:
                        self.hero_buttons_active[i] = {"action": "Bet", "pos": (x, y)}
                        any_button_active = True
                    elif any(keyword in cleaned_button_text for keyword in ["Fold", "Call", "Check", "Resume", "Cash"]):
                        any_button_active = True
                        self.hero_buttons_active[i] = {"action": cleaned_button_text, "pos": (x, y)}

                    time.sleep(0.4)

        if any_button_active:
            if not self.action_processed:
                # print(f"{Fore.RED}-------------------------------------------------------")
                # print(f"{Fore.RED}self.hero_buttons_active = {self.hero_buttons_active}")
                # print(f"{Fore.RED}-------------------------------------------------------")

                if self.game_state.round_count > 0:

                    if self.game_state.current_board_stage == 'Pre-Flop':

                        hero_role = self.game_state.players[self.game_state.hero_player_number].get('role')
                        hero_cards = self.game_state.players[self.game_state.hero_player_number].get('cards')

                        print(F"{Fore.RED} HERO CARDS: {hero_cards}")

                        is_playable_card = False

                        if hero_cards:
                            is_playable_card = self.hero_hand_range.is_hand_in_range(hero_cards)

                        if is_playable_card:

                            analysis_thread = threading.Thread(target=self.analyze_and_log)
                            analysis_thread.start()
                            print(F"{Fore.GREEN} PLAYABLE CARD: {hero_cards} in {hero_role} ROLE")

                        else:
                            self.hero_action.execute_action(None, "Fold", None)
                            self.game_state.update_player(self.game_state.hero_player_number, action='Fold')

                            self.hero_info.update_action_count(self.game_state.round_count, self.game_state.players[
                                self.game_state.hero_player_number].get('role'),
                                                               self.game_state.current_board_stage,
                                                               'Fold')

                            print(F"{Fore.RED} UNPLAYABLE CARD: {hero_cards} in {hero_role} ROLE ")


                    else:
                        analysis_thread = threading.Thread(target=self.analyze_and_log)
                        analysis_thread.start()

                self.action_processed = True
        else:
            # If no buttons are active, reset the states
            self.hero_buttons_active = {}
            self.action_processed = False

    def analyze_and_log(self):

        action_result = self.poker_assistant.AnalyzeAI(self.hero_buttons_active, self.game_state.get_ai_log())

        print(f"{Fore.CYAN}self.poker_assistant.AnalyzeAI RESULT: {action_result}")

        if action_result is not None:
            self.game_state.add_log_entry({'method': 'update_hero_action',
                                           'Action': action_result['Action'],
                                           'Amount': action_result['Amount'],
                                           'Tactic': action_result['Tactic'],
                                           'Strategy': action_result['Strategy'],
                                           'Explanation': action_result['Explanation']
                                           })

            self.hero_info.add_strategy(action_result['Strategy'])
            self.hero_info.add_tactic(action_result['Tactic'])
            self.hero_info.update_action_count(self.game_state.round_count,
                                               self.game_state.players[self.game_state.hero_player_number].get('role'),
                                               self.game_state.current_board_stage,
                                               action_result['Action'])

    def image_hash(self, image):
        """Generate a hash for an image."""

        image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        avg = gray.mean()  # Compute average pixel value

        return ''.join('1' if x > avg else '0' for x in gray.flatten())  # Create a binary hash

    def has_image_changed(self, unique_id, image):
        """Check if the image has changed based on hash comparison and a threshold."""
        current_hash = self.image_hash(image)
        previous_hash = self.previous_hashes.get(unique_id)

        def hamming_distance(hash1, hash2):
            """Calculate the Hamming distance between two binary strings."""
            return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

        if previous_hash is not None:

            difference = hamming_distance(current_hash, previous_hash)

            # print(F"{Fore.RED} difference = {difference}")

            if difference > 100:
                self.previous_hashes[unique_id] = current_hash
                return True
        else:
            # If no previous hash, store the current one
            self.previous_hashes[unique_id] = current_hash

        return False

    def detect_text(self, relative_x, relative_y, width, height):
        """
        Detect text from a specified region of the screen with optimizations.
        """
        screenshot = self.capture_screen_area(relative_x, relative_y, width, height)

        if screenshot is None:
            return None

        screenshot_array = np.array(screenshot)
        gray_image = cv2.cvtColor(screenshot_array, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Custom configurations for Tesseract
        custom_config = r'--oem 3 --psm 6'
        detected_text = pytesseract.image_to_string(thresh_image, config=custom_config)

        return detected_text.strip()

    # def update_plot_impl(self, screenshot_array):
    #     ax.clear()
    #     ax.imshow(screenshot_array)
    #     canvas.draw_idle()
    # def update_plot_safely(self, screenshot_array):
    #     if root.winfo_exists():  # Check if window still exists
    #         root.after(0, lambda: update_plot_impl(screenshot_array))

    def _detect_text_changed_impl(self, player_number, unique_id, relative_x, relative_y, width, height, typeofthing=None):
        screenshot = self.capture_screen_area(relative_x, relative_y, width, height)
        if screenshot is None:
            return None

        screenshot_array = np.array(screenshot)
        if self.has_image_changed(unique_id, screenshot_array):
            gray_image = cv2.cvtColor(screenshot_array, cv2.COLOR_BGR2GRAY)
            _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            custom_config = r'--oem 3 --psm 6'
            detected_text = pytesseract.image_to_string(thresh_image, config=custom_config).strip()
            return (player_number, detected_text)  # Return player number along with the text
        return None

    def detect_hero_combination_name(self):
        """Detect the card combination name of the hero player."""
        # Coordinates and dimensions for the hero's card combination name area
        relative_x = 0.820
        relative_y = 0.722

        width = 162
        height = 30

        # Use the existing detect_text method to read the combination name
        hero_combination_name = self.detect_text(relative_x, relative_y, width, height)

        # Update self.hero_cards_combination only if the value has changed
        if hero_combination_name is not None and len(hero_combination_name) > 4:

            if hero_combination_name != self.game_state.hero_cards_combination:
                print(F"{Fore.YELLOW}hero_combination_name = {hero_combination_name}")

                self.game_state.hero_cards_combination = hero_combination_name
        else:
            return None

        return hero_combination_name

    def _detect_player_stack_and_action_impl(self, player_number):
        if self.last_action_player == player_number:
            return None

        player_regions = {
            1: {'stack': (0.467, 0.732), 'action': (0.467, 0.701)},  # DONE
            2: {'stack': (0.059, 0.560), 'action': (0.059, 0.529)},  # DONE
            3: {'stack': (0.093, 0.265), 'action': (0.093, 0.235)},  # DONE
            4: {'stack': (0.430, 0.173), 'action': (0.430, 0.144)},  # DONE
            5: {'stack': (0.814, 0.265), 'action': (0.814, 0.235)},  # DONE
            6: {'stack': (0.846, 0.560), 'action': (0.842, 0.530)}  # DONE
        }
        region_stack_x, region_stack_y = player_regions[player_number]['stack']
        region_action_x, region_action_y = player_regions[player_number]['action']
        width = 95  # ...
        height = 24  # ...

        detected_stack_text = self.detect_text_changed(player_number, player_number + 10, region_stack_x,
                                                       region_stack_y, width, height, typeofthing='stack')
        if detected_stack_text is None:
            detected_stack_text = ''

        self.update_player_active_state(player_number, detected_stack_text)  # This is okay here

        current_stack_size, stack_size_change = self.get_player_stack_size(player_number, detected_stack_text)

        detected_action_text = self.detect_text_changed(player_number, player_number + 20, region_action_x,
                                                        region_action_y, width, height)
        if detected_action_text is None:
            detected_action_text = ''

        detected_action = detected_action_text.lower()
        bet_amount = 0

        if current_stack_size is not None:
            if stack_size_change < 20:
                bet_amount = stack_size_change

        if detected_action == "fold":
            # print('nigga')
            self.game_state.update_player(player_number, action='Fold')
            # return
            # print(f"Player{player_number}: {detected_action_text}")

        elif detected_action == "resume":
            self.game_state.update_player(player_number, action='Resume')
            # return
            # print(f"Player{player_number}: {detected_action_text}")

        elif detected_action == "check":
            self.game_state.update_player(player_number, action='Check')
            # return
            # print(f"Player{player_number}: {detected_action_text}")
        elif detected_action == "call":
            self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount,
                                          action='Call')
            # return
            # print(f"Player{player_number}: {detected_action_text}")
        elif detected_action == 'raise':
            self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount,
                                          action='Raise')
            # return
            # print(f"Player{player_number}: {detected_action_text}")
        elif detected_action == "bet":
            self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount, action='Bet')
            # return
        elif "won" in detected_action:
            won_amount_number = self.get_won_amount(detected_action)
            self.game_state.update_player(player_number, stack_size=current_stack_size, won_amount=won_amount_number)
            # return
            # print(f"Player{player_number}: {detected_action_text}")

        return (player_number, current_stack_size, stack_size_change, detected_action)  # Return the data with player number

    def _detect_player_stack_and_action_gui(self, result):
        if result is not None:
            player_number, current_stack_size, stack_size_change, detected_action = result
            # Update GUI elements with the received data for the correct player
            # Example (replace with your actual GUI elements):
            # if player_number == 1:
            #     self.player_1_stack_label.config(text=str(current_stack_size))
            #     self.player_1_action_label.config(text=detected_action)
            pass  # Replace with your actual GUI update code

            if current_stack_size is not None:
                self.game_state.update_player(player_number, stack_size=current_stack_size)

            if detected_action == "fold":
                self.game_state.update_player(player_number, action='Fold')
            elif detected_action == "resume":
                self.game_state.update_player(player_number, action='Resume')
            elif detected_action == "check":
                self.game_state.update_player(player_number, action='Check')
            elif detected_action == "call":
                self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount,
                                              action='Call')
            elif detected_action == 'raise':
                self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount,action="Raise")
            elif detected_action == "bet":
                self.game_state.update_player(player_number, stack_size=current_stack_size, amount=bet_amount,
                                              action='Bet')
            elif "won" in detected_action:
                won_amount_number = self.get_won_amount(detected_action)
                self.game_state.update_player(player_number, stack_size=current_stack_size,
                                              won_amount=won_amount_number)


    def update_player_active_state(self, player_number, detected_stack_text):

        # Check for 'Sitting Out', 'Sitting', or 'SEAT' in the detected text
        if re.search(r'sitting|seat|disconnect', detected_stack_text, re.IGNORECASE):

            current_status = self.game_state.players.get(player_number, {}).get('status')

            if current_status == 'Active':
                # Update player status to inactive
                self.game_state.update_player(player_number, status='Inactive')
        else:
            self.game_state.update_player(player_number, status='Active')

    def get_won_amount(self, detected_text):
        # First, remove commas from the detected text
        detected_text = detected_text.replace(',', '')

        # Use regular expression to extract the numeric value (float) from the text
        match = re.search(r'\d+(\.\d+)?', detected_text)

        if match:
            # Convert the matched string to a floating point number
            won_amount = float(match.group())

            return won_amount
        return 0

    def get_player_stack_size(self, player_number, detected_text):
        """
        Parse the detected text for stack size and update the game state.
        """

        # First, remove commas from the detected text
        detected_text = detected_text.replace(',', '')

        # print("Stack size potential: ", detected_text)

        # Use regular expression to extract the numeric value (float) from the text
        match = re.search(r'\d+(\.\d+)?', detected_text)

        if match:

            # Convert the matched string to a floating point number
            current_stack_size = float(match.group())

            # Retrieve the previous stack size from the game state
            old_stack_size = self.game_state.players.get(player_number, {}).get('stack_size', 0.0)

            if old_stack_size == 0:
                self.game_state.update_player(player_number, stack_size=current_stack_size)
                return None, None

            # Update the game state only if there's a change in the stack size
            if old_stack_size != current_stack_size:
                # Calculate change in stack size
                stack_size_change = current_stack_size - old_stack_size

                if stack_size_change < 0:
                    stack_size_change = -stack_size_change

                # print(stack_size_change)
                return current_stack_size, stack_size_change

        # If no valid number is found, return None
        return None, None

    def detect_player_turn(self):
        """
        Loop through all players and detect which player's turn it is by checking the gray bar presence.

        Return the number of the active player.
        """

        # Player1 Turn: 0.578, 0.734
        # Player2 Turn: 0.049, 0.564
        # Player3 Turn: 0.084, 0.266
        # Player4 Turn: 0.428, 0.172
        # Player5 Turn: 0.916, 0.260
        # Player6 Turn: 0.947, 0.556

        gray_background_region = {
            1: (0.431, 0.782), #--good
            2: (0.06, 0.609), #--good
            3: (0.090, 0.316), #--good
            4: (0.431, 0.224), #--good
            5: (0.773, 0.316), #--good
            6: (0.804, 0.609) #--good
        }

        player_similarities = {}

        for player_number, (region_x, region_y) in gray_background_region.items():
            progress_bar = self.capture_screen_area(region_x, region_y, 132, 8)
            progress_bar_gray = cv2.cvtColor(np.array(progress_bar), cv2.COLOR_BGR2GRAY)

            is_turn = False
            similarity, _ = ssim(progress_bar_gray, self.progress_bar_template, full=True)

            player_similarities[player_number] = similarity

        # print(player_similarities)

        player_number = max(player_similarities, key=player_similarities.get, default=None)

        # print(player_number)

        with self.game_state_lock:

            if self.game_state.get_current_player_turn() != player_number:
                self.game_state.update_player(player_number, turn=True)

                self.last_active_player = player_number

            return player_number

        # # Return the last active player if no active player is detected
        # return self.last_active_player

    def is_gray_bar_present(self, x, y):
        try:
            # Check if the coordinates are within the screen bounds
            screen_width, screen_height = pyautogui.size()
            if 0 <= x < screen_width and 0 <= y < screen_height:
                # Capture the color of the pixel at the given coordinates
                pixel_color = pyautogui.pixel(x, y)
            else:
                print(f"Coordinates ({x}, {y}) are out of screen bounds.")
                return False

            # 858789 = gray bar
            # 133 = R
            # 135 = G
            # 137 = B

            # Define the color range that indicates a gray bar
            gray_lower = np.array([130, 130, 130], dtype="uint8")
            gray_upper = np.array([140, 140, 140], dtype="uint8")

            # Check if the pixel color falls within the green range
            in_range = all(gray_lower[i] <= pixel_color[i] <= gray_upper[i] for i in range(3))
            return in_range

        except Exception as e:
            print(f"Error in gray bar at ({x}, {y}): {e}")
            return False

    def detect_total_pot_size(self):
        """
        Detect the total pot size on the table and update the game state only if it has changed.
        """
        detected_text = self.detect_text(0.445, 0.312, 110, 28)  # Coordinates for the total pot size

        # Check if detected_text is None
        if detected_text is None:
            return

        try:
            # Use regular expression to extract the numeric value from the text
            match = re.search(r'[\d,.]+', detected_text)
            if match:
                # Convert the matched string to a floating point number
                pot_size_str = match.group().replace(",", "")
                new_pot_size = float(pot_size_str)

                # Check if the detected pot size is different from the current pot size in the game state
                if new_pot_size != self.game_state.total_pot:
                    with self.game_state_lock:
                        self.game_state.update_total_pot(new_pot_size)
                        # print(f"Total Pot Size updated to: {new_pot_size}")
                # else:
                # print("No change in Total Pot Size.")
            # else:
            # print(f"No valid pot size found in detected text: '{detected_text}'")

        except ValueError:
            # Handle cases where the extracted text is not a valid number
            print(f"Unable to parse total pot size from detected text: '{detected_text}'")
        return detected_text

    def is_color_active(self, x, y, tolerance=30):
        """
        Check if a card is placed at the given coordinates by detecting the color.
        :param x: The x-coordinate of the pixel.
        :param y: The y-coordinate of the pixel.
        :param tolerance: The color tolerance. Default is 10.
        :return: True if a card color is detected, False otherwise.
        """
        # Capture the color of the pixel at the given coordinates
        pixel_color = pyautogui.pixel(x, y)

        # Define the color range that indicates a card is placed (range of white)
        min_color = (255 - tolerance, 255 - tolerance, 255 - tolerance)
        max_color = (255, 255, 255)

        # Check if the pixel color is within the specified range
        return all(min_color[i] <= pixel_color[i] <= max_color[i] for i in range(3))

    def check_player_card_active(self, player_number):
        """
        Check if both cards of a specified player have white pixels at given coordinates.
        """
        # Player1 Cards1: 0.450, 0.687
        # Player1 Cards2: 0.512, 0.687

        # Player2 Cards1: 0.075, 0.514
        # Player2 Cards2: 0.136, 0.513

        # Player3 Cards1: 0.106, 0.218
        # Player3 Cards2: 0.168, 0.218

        # Player4 Cards1: 0.450, 0.126
        # Player4 Cards2: 0.512, 0.126

        # Player5 Cards1: 0.792, 0.217
        # Player5 Cards2: 0.855, 0.217

        # Player6 Cards1: 0.825, 0.513
        # Player6 Cards2: 0.886, 0.514

        # Relative coordinates for each player's cards
        relative_coordinates = {
            1: [(0.450, 0.687), (0.512, 0.687)],
            2: [(0.075, 0.513), (0.136, 0.513)],
            3: [(0.106, 0.218), (0.168, 0.218)],
            4: [(0.450, 0.126), (0.512, 0.126)],
            5: [(0.792, 0.217), (0.855, 0.217)],
            6: [(0.825, 0.513), (0.886, 0.514)]
        }

        # Check for each card of the player
        for rel_x, rel_y in relative_coordinates[player_number]:

            screen_x, screen_y = self.convert_to_screen_coords(rel_x, rel_y)
            pixel_color = pyautogui.pixel(screen_x, screen_y)

            if not self.is_pixel_white(pixel_color):
                return False

        return True

    def find_player_cards(self, player_number):
        """Find the cards of a specific player."""
        # Player Card Numbers/Letters
        # Player Cards1: 0.442, 0.621  Player Cards: x=0.442, y=0.655
        # Player Cards2: 0.505, 0.621

        # Player Cards1: 0.066, 0.448
        # Player Cards2: 0.130, 0.448

        # Player Cards1: 0.099, 0.153
        # Player Cards2: 0.161, 0.153

        # Player Cards1: 0.442, 0.062
        # Player Cards2: 0.505, 0.062

        # Player Cards1: 0.785, 0.153
        # Player Cards2: 0.847, 0.153

        # Player Cards1: 0.816, 0.448
        # Player Cards2: 0.879, 0.448

        # Coordinates for all players (Number Position, Icon Position)
        player_card_positions = {
            1: [(0.442, 0.621), (0.501, 0.621)],  # Player 1
            2: [(0.064, 0.448), (0.128, 0.448)],  # Player 2
            3: [(0.097, 0.153), (0.159, 0.153)],  # Player 3
            4: [(0.439, 0.062), (0.502, 0.062)],  # Player 4
            5: [(0.782, 0.153), (0.844, 0.153)],  # Player 5
            6: [(0.813, 0.448), (0.876, 0.448)]  # Player 6
        }

        if player_number not in player_card_positions:
            print(f"Invalid player number: {player_number}")
            return None

        # print(f"Processing cards for Player {player_number}")
        cards_found = []

        for index, (num_x, num_y) in enumerate(player_card_positions[player_number], start=1):
            icon_x, icon_y = num_x, num_y + 0.032  # Adjust for icon position
            icon_width, icon_height = 30, 30  # Card icon dimensions
            num_width, num_height = 27, 50 # Card number/letter dimensions

            # Process card suit
            card_filename = f"card{index}Icon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            screenshot_icon = self.capture_screen_area(icon_x, icon_y, icon_width,
                                                       icon_height)  # , filename=card_filename)
            screenshot_icon_gray = cv2.cvtColor(np.array(screenshot_icon), cv2.COLOR_BGR2GRAY)

            # plt.imshow(screenshot_icon_gray)
            # plt.show()

            print('yo')

            card_suit = None

            for suit, template in self.card_icon_templates.items():

                # print(screenshot_icon_gray.shape)
                # print(template.shape)

                similarity, _ = ssim(screenshot_icon_gray, template, full=True)
                if similarity > 0.6:
                    card_suit = suit
                    # print(f"(Icon) for Player {player_number}, Card {index}: {card_suit}")

            # Capture and process card number/letter using template matching
            card2_filename = f"card{index}Letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            screenshot_num = self.capture_screen_area(num_x, num_y, num_width, num_height)  # , filename=card2_filename)
            screenshot_num_gray = cv2.cvtColor(np.array(screenshot_num), cv2.COLOR_BGR2GRAY)

            card_rank = None
            for rank, template in self.card_number_templates.items():
                similarity, _ = ssim(screenshot_num_gray, template, full=True)

                # print(f"rank: {rank}, similarity: {similarity}")

                if similarity > 0.6:
                    card_rank = rank
                    # print(f"(Rank) for Player {player_number}, Card {index}: {card_rank}")

            if card_rank and card_suit:
                card = f'{card_rank}{card_suit}'
                cards_found.append(card)
            else:
                cards_found.append(None)

        valid_cards = [card for card in cards_found if card is not None]

        # if valid_cards:
        # print(f"Valid cards for Player {player_number}: {valid_cards}")

        return valid_cards

    def find_cards_on_table(self):
        card_number_positions = [
            (0.366, 0.352),  # Card 1 position coordinates x,y
            (0.429, 0.352),  # Card 2 position coordinates x,y
            (0.494, 0.352),  # Card 3 position coordinates x,y
            (0.558, 0.352),  # Card 4 position coordinates x,y
            (0.624, 0.352)  # Card 5 position coordinates x,y
        ]

        card_icon_positions = [
            (0.361, 0.421),  # Card 1 position coordinates x,y
            (0.425, 0.421),  # Card 2 position coordinates x,y
            (0.490, 0.421),  # Card 3 position coordinates x,y
            (0.554, 0.421),  # Card 4 position coordinates x,y
            (0.620, 0.421)  # Card 5 position coordinates x,y
        ]

        cards_found = []

        for index, (icon_position, number_position) in enumerate(zip(card_icon_positions, card_number_positions),
                                                                 start=1):
            # print(f"Card {index}")

            x, y = icon_position
            num_x, num_y = number_position
            icon_width, icon_height = 30, 30# Card icon dimensions
            num_width, num_height = 27, 50  # Card number/letter dimensions

            # Capture and process card suit
            screenshot_icon = self.capture_screen_area(x, y, icon_width, icon_height)

            # try:
            screenshot_icon_gray = cv2.cvtColor(np.array(screenshot_icon), cv2.COLOR_BGR2GRAY)

            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = f"{index}_screenshot_{timestamp}.png"
            # filepath = os.path.join("card_images", filename)
            #
            # # Save the image
            # plt.imsave(filepath, screenshot_icon_gray, cmap='gray')

            #
            # plt.imshow(screenshot_icon_gray)
            # plt.show()

            # except:
            # print(np.array(screenshot_icon))

            # plt.imshow(screenshot_icon)
            # plt.show()

            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = f"screenshot_{timestamp}.png"
            # filepath = os.path.join("card_images", filename)
            #
            # # Save the image
            # plt.imsave(filepath, screenshot_icon_gray, cmap='gray')

            possible_suits = {}
            for card_name, template in self.card_icon_templates.items():
                similarity, _ = ssim(screenshot_icon_gray, template, full=True)

                # print(f'Suit : {card_name}, Similarity: {similarity}')

                if similarity > 0.3:
                    possible_suits[card_name] = similarity

            card_suit = max(possible_suits, key=possible_suits.get, default=None)

            # Capture and process card number/letter using template matching
            screenshot_num = self.capture_screen_area(num_x, num_y, num_width, num_height)
            screenshot_num_gray = cv2.cvtColor(np.array(screenshot_num), cv2.COLOR_BGR2GRAY)

            # print('hey')
            # plt.imshow(screenshot_num_gray)
            # plt.show()

            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = f"hi{index}_screenshot_{timestamp}.png"
            # filepath = os.path.join("card_images", filename)
            #
            # # Save the image
            # plt.imsave(filepath, screenshot_num_gray, cmap='gray')

            card_rank = None
            possible_ranks = {}
            for rank, template in self.card_number_templates.items():
                similarity, _ = ssim(screenshot_num_gray, template, full=True)

                # print(f"Rank : {rank}, Similarity: {similarity}")

                if similarity > 0.3:
                    possible_ranks[rank] = similarity

            card_rank = max(possible_ranks, key=possible_ranks.get, default=None)

            if card_rank and card_suit:
                card = f'{card_rank}{card_suit}'
                cards_found.append(card)
            else:
                cards_found.append(None)

        # After processing all cards, filter out None values
        valid_cards_found = [card for card in cards_found if card is not None]

        # valid_cards_count = len(valid_cards_found)

        with self.game_state_lock:

            # Check if no cards are found on the table
            if not valid_cards_found:
                if self.cards_on_table:
                    self.cards_on_table = False
                    self.game_state.update_community_cards([])
                    self.last_detected_cards = []  # Reset the last detected cards
                    # print("New round started: Community cards cleared.")
            else:
                self.cards_on_table = True

                # Update community cards if there is a change, and the number of detected cards
                # is not less than the number of cards in the last detected state
                if valid_cards_found != self.last_detected_cards and len(valid_cards_found) >= len(
                        self.last_detected_cards):
                    self.game_state.update_community_cards(valid_cards_found)
                    self.last_detected_cards = valid_cards_found

        return valid_cards_found

    def convert_to_screen_coords(self, rel_x, rel_y):
        """Convert relative coordinates to screen coordinates based on the window position."""
        # Calculate absolute screen coordinates
        abs_x = self.window.left + int(rel_x * self.window.width)
        abs_y = self.window.top + int(rel_y * self.window.height)
        return abs_x, abs_y

    def find_dealer_button(self, button_template):
        """Find the dealer button in one of the defined regions."""

        dealer_button_regions = {
            1: (0.392, 0.611),  # done - these motherfuckers change positions randomly -- good
            2: (0.198, 0.463),  # done - these motherfuckers change positions randomly -- good
            3: (0.212, 0.343),  # done - these motherfuckers change positions randomly
            4: (0.529, 0.258),  # done - these motherfuckers change positions randomly -- good
            5: (0.769, 0.336),  # done - these motherfuckers change positions randomly -- good
            6: (0.720, 0.560)  # done - these motherfuckers change positions randomly -- good
        }

        for player_number, (region_x, region_y) in dealer_button_regions.items():

            width, height = 30,24 # Adjust these dimensions as necessary
            screenshot = self.capture_screen_area(region_x, region_y, width, height)
            screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)

            # plt.imshow(screenshot_gray)
            # plt.show()
            #
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = f"screenshot_{timestamp}.png"
            # filepath = os.path.join("stupid", filename)
            #
            # # Save the image
            # plt.imsave(filepath, screenshot_gray, cmap='gray')

            similarity, _ = ssim(screenshot_gray, button_template, full=True)

            # plt.imshow(screenshot_gray)
            # plt.title("Potential Dealer Button")
            # plt.show()
            #
            # plt.imshow(button_template)
            # plt.title("Button Template")
            # plt.show()

            # print(f"player number: {player_number}, score: {similarity}")

            threshold = 0.2  # Adjust based on testing

            if similarity > threshold:
                # print("Dealer buttons active")

                # Check if the detected dealer position is different from the current one
                if self.game_state.dealer_position != player_number:

                    with self.game_state_lock:

                        self.game_state.update_dealer_position(player_number)

                        self.game_state.dealer_position = player_number

                        if self.game_state.round_count > 1:
                            if self.game_state.round_count % 12 == 0:  # Do every 8 rounds
                                # Start a new thread for player analysis
                                analysis_thread = threading.Thread(target=self.poker_assistant.analyze_players_gpt4,
                                                                   args=(self.game_state.all_round_logs,))
                                analysis_thread.start()

                        self.game_state.reset_for_new_round()  # THIS MUST BE AFTER analyze_players_gpt4() function so it doesn reset data before Analysis!

                    # print(f"game_state.dealer_position = {player_number}")

                return player_number  # Dealer button found at this player

        return None  # Dealer button not found

    def _player_cards(self):
        for player_number in range(1, 7):
            if self.check_player_card_active(player_number):
                self.image_processing_queue.put(("find_player_cards", (player_number,)))
        time.sleep(1)


    def _table_cards(self):
        self.image_processing_queue.put(("find_cards_on_table", ()))  # Queue the task
        time.sleep(0.7)  # Sleep after queuing, not during processing

    def _player_turn(self):
        self.image_processing_queue.put(("detect_player_turn", ()))  # Queue the task
        time.sleep(0.1)

    def _dealer_button(self):
        self.image_processing_queue.put(("find_dealer_button", (self.dealer_button_image,)))
        time.sleep(0.8)

    def _player_actions(self):
        current_time = time.time()
        for player_number in range(1, 7):  # Iterate through all players
            if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:
                self.image_processing_queue.put(("detect_player_stack_and_action", (player_number,)))  # Queue the task
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number
        time.sleep(0.05)  # Short sleep is okay here

# def continuous_detection_player_action2(self):  # Example for player 1, apply to others

        current_time = time.time()
        player_number = 2  # Define the player number here
        if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:  # Adjust delay as needed
            detected_action = self.detect_player_stack_and_action(player_number)
            if detected_action:  # Check if an action was actually detected
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number  # Update last action player
        time.sleep(0.05)

# def continuous_detection_player_action3(self):  # Example for player 1, apply to others

        current_time = time.time()
        player_number = 3  # Define the player number here
        if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:  # Adjust delay as needed
            detected_action = self.detect_player_stack_and_action(player_number)
            if detected_action:  # Check if an action was actually detected
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number  # Update last action player
        time.sleep(0.05)

# def continuous_detection_player_action4(self):  # Example for player 1, apply to others

        current_time = time.time()
        player_number = 4  # Define the player number here
        if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:  # Adjust delay as needed
            detected_action = self.detect_player_stack_and_action(player_number)
            if detected_action:  # Check if an action was actually detected
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number  # Update last action player
        time.sleep(0.05)

# def continuous_detection_player_action5(self):  # Example for player 1, apply to others

        current_time = time.time()
        player_number = 5  # Define the player number here
        if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:  # Adjust delay as needed
            detected_action = self.detect_player_stack_and_action(player_number)
            if detected_action:  # Check if an action was actually detected
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number  # Update last action player
        time.sleep(0.05)

# def continuous_detection_player_action6(self):  # Example for player 1, apply to others

        current_time = time.time()
        player_number = 6  # Define the player number here
        if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 1:  # Adjust delay as needed
            detected_action = self.detect_player_stack_and_action(player_number)
            if detected_action:  # Check if an action was actually detected
                self.last_action_time[player_number] = current_time
                self.last_action_player = player_number  # Update last action player
        time.sleep(0.05)

    def _total_pot_size(self):
        self.image_processing_queue.put(("detect_total_pot_size", ()))  # Queue the task
        time.sleep(0.6)

    def _hero_action_buttons(self):
        self.image_processing_queue.put(("detect_hero_buttons", ()))  # Queue the task
        time.sleep(0.3)


            # except Exception as e:
            # print(f"Error in continuous detection: {e}")

    def initiate_shutdown(self):
        """Initiate the shutdown process."""

        print("Shutdown initiated...")
        self.shutdown_flag.set()  # Signal all threads to shutdown
        self.shutdown()  # Call the shutdown method to gracefully close threads

    def shutdown(self):
        """Shut down all threads gracefully."""

        print("Shutting down threads...")

        print(f"Active threads at the beginning of shutdown: {threading.active_count()}")

        for thread in self.threads:
            if thread.is_alive():
                self.shutdown_flag.set()  # Signal the thread to shutdown
                thread.join()  # Wait for the thread to finish
                print(f"Active threads after joining a thread: {threading.active_count()}")

        print("All threads have been joined.")
        keyboard.unhook_all()

    def start_continuous_detection(self):  # No overlay creation here
        """Starts continuous detection of game state changes."""
        self.root = tk.Tk()  # Create the main window
        self.root.withdraw()  # Hide the main window

        for task_name in self.update_intervals:
            self.schedule_update(task_name)

        self.root.mainloop()  # Start the Tkinter event loop

    def schedule_update(self, task_name):
        """Schedules a specific update function using after()."""
        interval = self.update_intervals[task_name]
        task_func = getattr(self, f"_{task_name}")  # Get function by name

        def run_task_and_reschedule():
            task_func()  # Run the task (NO WHILE LOOP INSIDE)
            self.root.after(interval, run_task_and_reschedule)  # Reschedule

        self.root.after(interval, run_task_and_reschedule)  # First call

    def start_image_processing_thread(self):
        self.image_processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.image_processing_thread.start()

    def process_images(self):
        while not self.shutdown_flag.is_set():
            try:
                task = self.image_processing_queue.get(timeout=1)  # Check shutdown flag
                if task is None:
                    break

                method_name, args = task
                try:  # Add a try/except block here
                    result = getattr(self, f"_{method_name}_impl")(*args)
                    if result is not None:
                        self.gui_update_queue.put((method_name, result))
                except Exception as e:
                    print(f"Error in image processing for {method_name}: {e}")
                finally:
                    self.image_processing_queue.task_done()
            except queue.Empty:
                pass

    def start_gui_update_thread(self):
        self.gui_update_thread = threading.Thread(target=self.update_gui, daemon=True)
        self.gui_update_thread.start()

    def update_gui(self):
        while not self.shutdown_flag.is_set():
            try:
                update_data = self.gui_update_queue.get(timeout=1)
                if update_data is None:
                    break

                method_name, result = update_data
                try:  # Add a try/except block here
                    self.root.after(0, lambda: getattr(self, f"_{method_name}_gui")(result))  # Update on main thread
                except Exception as e:
                    print(f"Error in GUI update for {method_name}: {e}")
                finally:
                    self.gui_update_queue.task_done()
            except queue.Empty:
                pass