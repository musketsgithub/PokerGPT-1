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

        self.image_dict = {}  # Dictionary to store previous hashes for each player

        self.detected_text_dict = {}

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

        self.similarities = []

        self.pending_actions = {}  # Store actions that are waiting for stack updates
        self.action_timeout = 2.0  # Timeout for pending actions in seconds
        self.last_stack_updates = {}

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

    # def image_hash(self, image):
    #     """Generate a hash for an image."""
    #
    #     image = np.array(image)
    #
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #
    #     avg = gray.mean()  # Compute average pixel value
    #
    #     return ''.join('1' if x > avg else '0' for x in gray.flatten())  # Create a binary hash

    # def has_image_changed(self, player_number, image):
    #     """Check if the image has changed based on similarity comparison."""
    #     current_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    #
    #     previous_image = self.image_dict.get(player_number)
    #
    #     if previous_image is not None:
    #         try:
    #             similarity, _ = ssim(current_image, previous_image, full=True)
    #
    #             # self.similarities.append(similarity)
    #             # plt.hist(self.similarities)
    #             # plt.show()
    #             # print('gangbang')
    #
    #             # print(f"[DEBUG] Player {player_number} similarity: {similarity}")
    #
    #             if similarity < 0.32:
    #                 print(f"[DEBUG] Change detected for player {player_number}, updating image dict")
    #                 self.image_dict[player_number] = current_image.copy()  # Make sure to copy the image
    #                 return True
    #
    #             print(f"[DEBUG] No change detected for player {player_number}")
    #             return False
    #
    #         except Exception as e:
    #             print(f"[DEBUG] Error comparing images for player {player_number}: {e}")
    #             return False
    #     else:
    #         print(f"[DEBUG] First image for player {player_number}, storing in dict")
    #         self.image_dict[player_number] = current_image.copy()  # Make sure to copy the image
    #         return True

    def detect_text(self, relative_x, relative_y, width, height):
        """
        Detect text from a specified region of the screen with optimizations.
        """
        screenshot = self.capture_screen_area(relative_x, relative_y, width, height, resize_dimensions=(width, height))

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

        if detected_text is None:  # Handle the None case right at the beginning
            return None, None

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
            progress_bar = self.capture_screen_area(region_x, region_y, 132, 8, resize_dimensions=(132,8))
            progress_bar_gray = cv2.cvtColor(np.array(progress_bar), cv2.COLOR_BGR2GRAY)

            is_turn = False
            # print("Progress bar shape", progress_bar_gray.shape)
            similarity, _ = ssim(progress_bar_gray, self.progress_bar_template, full=True)

            player_similarities[player_number] = similarity

        # print(player_similarities)

        player_number = max(player_similarities, key=player_similarities.get, default=None)

        # print(player_number)

        with self.game_state_lock:
            if self.game_state.get_current_player_turn() != player_number:
                self.game_state.update_player(player_number, turn=True)
                self.last_active_player = player_number

                # Check if it's the hero's turn
                if player_number == self.game_state.hero_player_number:
                    self.game_state.analyze_hand(self.poker_assistant.openai_client)  # Pass openai_client here

            return player_number

        # # Return the last active player if no active player is detected
        # return self.last_active_player

    def detect_total_pot_size(self):
        """
        Detect the total pot size on the table and update the game state only if it has changed.
        """
        detected_text = self.detect_text(0.445, 0.312, 110, 28)  # Coordinates for the total pot size

        # Check if detected_text is None
        if detected_text is None:
            return

        try:
            # Remove any commas and the "Pot: $" prefix
            cleaned_text = detected_text.replace(',', '').replace('Pot:', '').replace('$', '').replace('.', '').strip()

            # Convert to float, dividing by 100 to handle cents
            new_pot_size = float(cleaned_text) / 100

            # Check if the detected pot size is different from the current pot size in the game state
            if new_pot_size != self.game_state.total_pot:
                with self.game_state_lock:
                    self.game_state.update_total_pot(new_pot_size)
                    # print(f"Total Pot Size updated to: ${new_pot_size:.2f}")

        except ValueError as e:
            print(f"Unable to parse total pot size from detected text: '{detected_text}' (Error: {e})")
        except Exception as e:
            print(f"Unexpected error processing pot size: '{detected_text}' (Error: {e})")

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
        print('hello2')
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
            num_width, num_height = 200, 200 # Card number/letter dimensions

            # # Process card suit
            # card_filename = f"card{index}Icon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            #
            # screenshot_icon = self.capture_screen_area(icon_x, icon_y, icon_width,
            #                                            icon_height, resize_dimensions=(icon_width, icon_height))  # , filename=card_filename)
            # screenshot_icon_gray = cv2.cvtColor(np.array(screenshot_icon), cv2.COLOR_BGR2GRAY)
            #
            # card_suit = None
            #
            # for suit, template in self.card_icon_templates.items():
            #
            #     # print(screenshot_icon_gray.shape)
            #     # print(template.shape)
            #
            #     print("icon shape", screenshot_icon_gray.shape)
            #     print("template shape", template.shape)
            #     similarity, _ = ssim(screenshot_icon_gray, template, full=True)
            #     if similarity > 0.6:
            #         card_suit = suit
            #         # print(f"(Icon) for Player {player_number}, Card {index}: {card_suit}")
            #
            # # Capture and process card number/letter using template matching
            # card2_filename = f"card{index}Letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            if player_number == 1 and index == 1:
                screenshot_num = self.capture_screen_area(num_x, num_y, num_width, num_height, resize_dimensions=(num_width, num_height))  # , filename=card2_filename)
                screenshot_num_gray = cv2.cvtColor(np.array(screenshot_num), cv2.COLOR_BGR2GRAY)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join("card_images", filename)

                # Save the image
                plt.imsave(filepath, screenshot_num_gray, cmap='gray')

            # card_rank = None
            # for rank, template in self.card_number_templates.items():
            #     print(screenshot_num_gray.shape)
            #     print(template.shape)
            #     similarity, _ = ssim(screenshot_num_gray, template, full=True)
            #
            #     # print(f"rank: {rank}, similarity: {similarity}")
            #
            #     if similarity > 0.6:
            #         card_rank = rank
            #         # print(f"(Rank) for Player {player_number}, Card {index}: {card_rank}")
            #
            # if card_rank and card_suit:
            #     card = f'{card_rank}{card_suit}'
            #     cards_found.append(card)
            # else:
            #     cards_found.append(None)

        valid_cards = [card for card in cards_found if card is not None]

        # if valid_cards:
        # print(f"Valid cards for Player {player_number}: {valid_cards}")

        # print(valid_cards)
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
            screenshot_icon = self.capture_screen_area(x, y, icon_width, icon_height, resize_dimensions=(icon_width, icon_height))

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
            screenshot_num = self.capture_screen_area(num_x, num_y, num_width, num_height, resize_dimensions=(num_width, num_height))
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
            screenshot = self.capture_screen_area(region_x, region_y, width, height, resize_dimensions=(width, height))
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

    def action1(self):
        while True:
            player_number = 1
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
            self.detect_player_stack_and_action(1)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here

    def action2(self):
        while True:
            player_number = 2
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
            self.detect_player_stack_and_action(2)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here


    def action3(self):
        while True:
            player_number = 3
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
            self.detect_player_stack_and_action(3)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here

    def action4(self):
        while True:
            player_number = 4
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
            self.detect_player_stack_and_action(4)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here

    def action5(self):
        while True:
            player_number = 5
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
            self.detect_player_stack_and_action(5)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here

    def action6(self):
        marker=0
        while True:
            player_number = 6
            current_time = time.time()
            # if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] >  2:
            self.detect_player_stack_and_action(6)
            self.last_action_time[player_number] = current_time
            self.last_action_player = player_number
            time.sleep(0.01)  # Short sleep is okay here



    # def player_actions(self):
    #     while True:
    #         player_number = 1
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
    #             self.detect_player_stack_and_action(1)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here
    #
    #         player_number = 2
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
    #             self.detect_player_stack_and_action(2)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here
    #
    #         player_number = 3
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
    #             self.detect_player_stack_and_action(3)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here
    #
    #         player_number = 4
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
    #             self.detect_player_stack_and_action(4)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here
    #
    #         player_number = 5
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] > 2:
    #             self.detect_player_stack_and_action(5)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here
    #
    #         player_number = 6
    #         current_time = time.time()
    #         if player_number not in self.last_action_time or current_time - self.last_action_time[player_number] >  2:
    #             self.detect_player_stack_and_action(6)
    #             self.last_action_time[player_number] = current_time
    #             self.last_action_player = player_number
    #         time.sleep(0.05)  # Short sleep is okay here

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

    def start_continuous_detection(self):
        """Start the continuous detection in a new thread."""
        # #
        action_detection_thread1 = threading.Thread(target=self.action1)

        action_detection_thread2 = threading.Thread(target=self.action2)

        action_detection_thread3 = threading.Thread(target=self.action3)

        action_detection_thread4 = threading.Thread(target=self.action4)

        action_detection_thread5 = threading.Thread(target=self.action5)

        action_detection_thread6 = threading.Thread(target=self.action6)

        # player_action_thread = threading.Thread(target=self.player_actions)

        turn_detection_thread = threading.Thread(target=self.continuous_detection_player_turn)

        dealer_detection_thread = threading.Thread(target=self.continuous_detection_dealer_button)

        cards_detection_thread = threading.Thread(target=self.continuous_detection_table_cards)

        player_cards_detection_thread = threading.Thread(target=self.continuous_detection_player_cards)

        pot_detection_thread = threading.Thread(target=self.continuous_detection_total_pot_size)

        hero_buttons_thread = threading.Thread(target=self.continuous_detection_hero_action_buttons)

        action_detection_thread1.start()
        action_detection_thread2.start()
        action_detection_thread3.start()
        action_detection_thread4.start()
        action_detection_thread5.start()
        action_detection_thread6.start()

        # player_action_thread.start()

        turn_detection_thread.start()
        dealer_detection_thread.start()
        cards_detection_thread.start()
        player_cards_detection_thread.start()
        pot_detection_thread.start()
        hero_buttons_thread.start()

    def detect_player_stack_and_action(self, player_number):
        """Detect and process player stack size and actions with timing handling."""
        current_time = time.time()

        # Clean up expired pending actions
        if player_number in self.pending_actions:
            if current_time - self.pending_actions[player_number]['time'] > self.action_timeout:
                del self.pending_actions[player_number]

        # Define regions for all players
        player_regions = {
            1: {'stack': (0.467, 0.732), 'action': (0.467, 0.701)},
            2: {'stack': (0.059, 0.560), 'action': (0.059, 0.529)},
            3: {'stack': (0.093, 0.265), 'action': (0.093, 0.235)},
            4: {'stack': (0.430, 0.173), 'action': (0.430, 0.144)},
            5: {'stack': (0.814, 0.265), 'action': (0.814, 0.235)},
            6: {'stack': (0.846, 0.560), 'action': (0.842, 0.530)}
        }

        region_stack_x, region_stack_y = player_regions[player_number]['stack']
        region_action_x, region_action_y = player_regions[player_number]['action']
        width = 95
        height = 24

        # Detect stack size
        detected_stack_text = self.detect_text_changed(
            player_number,
            player_number + 10,
            region_stack_x,
            region_stack_y,
            width,
            height,
            typeofthing='stack'
        )

        # if player_number==1:
        #     print(detected_stack_text)

        # Process stack update
        if detected_stack_text is not None:
            with self.game_state_lock:
                self.update_player_active_state(player_number, detected_stack_text)
                current_stack_size, stack_size_change = self.get_player_stack_size(player_number, detected_stack_text)

                if current_stack_size is not None:
                    self.game_state.update_player(player_number, stack_size=current_stack_size)
                    self.last_stack_updates[player_number] = {
                        'time': current_time,
                        'change': stack_size_change
                    }

                    # Check if we have a pending action waiting for this stack update
                    if player_number in self.pending_actions:
                        pending = self.pending_actions[player_number]
                        if pending['action'] in ['call', 'raise', 'bet']:
                            self.game_state.update_player(
                                player_number,
                                stack_size=current_stack_size,
                                amount=stack_size_change if stack_size_change else 0,
                                action=pending['action'].capitalize()
                            )
                        del self.pending_actions[player_number]
                        self.last_action_player = player_number
                        return

        # Detect action
        detected_action_text = self.detect_text_changed(
            player_number,
            player_number + 20,
            region_action_x,
            region_action_y,
            width,
            height,
            typeofthing='action'
        )

        if detected_action_text is not None:
            detected_action = detected_action_text.lower()

            with self.game_state_lock:
                # For actions that don't require stack changes
                if detected_action in ['fold', 'resume', 'check']:
                    self.game_state.update_player(player_number, action=detected_action.capitalize())
                    self.last_action_player = player_number
                    return

                # For betting actions, check if we recently saw a stack change
                elif detected_action in ['call', 'raise', 'bet']:
                    recent_stack = self.last_stack_updates.get(player_number)
                    if recent_stack and (current_time - recent_stack['time'] < self.action_timeout):
                        # We have a recent stack change, use it
                        self.game_state.update_player(
                            player_number,
                            amount=recent_stack['change'] if recent_stack['change'] else 0,
                            action=detected_action.capitalize()
                        )
                        self.last_action_player = player_number
                    else:
                        # Store this action as pending
                        self.pending_actions[player_number] = {
                            'action': detected_action,
                            'time': current_time
                        }
                    return

                # For winning, which might come before stack update
                elif "won" in detected_action:
                    won_amount = self.get_won_amount(detected_action)
                    self.pending_actions[player_number] = {
                        'action': 'won',
                        'amount': won_amount,
                        'time': current_time
                    }
                    return

    def detect_text_changed(self, player_number, unique_id, relative_x, relative_y, width, height, typeofthing=None):
        """Detect text from a specified region of the screen with optimizations."""
        screenshot = self.capture_screen_area(relative_x, relative_y, width, height, resize_dimensions=(width, height))

        if screenshot is None:
            return None

        screenshot_array = np.array(screenshot)
        gray_image = cv2.cvtColor(screenshot_array, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Custom configurations for Tesseract
        custom_config = r'--oem 3 --psm 6'
        detected_text = pytesseract.image_to_string(thresh_image, config=custom_config)

        # Create a unique dictionary key for each type of text
        dict_key = f"player_{player_number}_{typeofthing}"

        # if player_number == 1:  # Debug logging for player 1
        #     if dict_key in self.detected_text_dict:
        #         print(f"Old Text ({typeofthing})", self.detected_text_dict[dict_key])
        #     print(f"New Text ({typeofthing})", detected_text.strip())

        # Check if text has changed
        if dict_key in self.detected_text_dict:
            if detected_text.strip() != self.detected_text_dict[dict_key]:
                self.detected_text_dict[dict_key] = detected_text.strip()
                return detected_text.strip()
            return None
        else:
            self.detected_text_dict[dict_key] = detected_text.strip()
            return None

    def update_player_active_state(self, player_number, detected_stack_text):

        if detected_stack_text is not None:  # Check if detected_stack_text is NOT None
            if re.search(r'sitting|seat|disconnect', detected_stack_text, re.IGNORECASE):
                current_status = self.game_state.players.get(player_number, {}).get('status')
                if current_status == 'Active':
                    self.game_state.update_player(player_number, status='Inactive')
            else:
                self.game_state.update_player(player_number, status='Active')
        #else: if it is None, do nothing, the player is already considered inactive

    def continuous_detection_player_turn(self):
        """Continuously detect game state changes."""
        while True:
            # try:
            if not self.window:
                time.sleep(0.2)  # Adjust the sleep time as needed
                continue  # Skip to the next iteration of the loop

            # Detect active players turn
            self.detect_player_turn()

            # Sleep before the next detection cycle
            time.sleep(0.1)  # Adjust the sleep time as needed

            # except Exception as e:
            # print(f"Error in continuous detection: {e}")

    def continuous_detection_dealer_button(self):
        """Continuously detect game state changes."""

        while True:
            if not self.window:
                time.sleep(0.2)  # Adjust the sleep time as needed
                continue  # Skip to the next iteration of the loop

            self.find_dealer_button(self.dealer_button_image)

            # Sleep before the next detection cycle
            time.sleep(0.8)  # Adjust the sleep time as needed

    def continuous_detection_table_cards(self):
        """Continuously detect game state changes."""
        while True:
            if not self.window:
                time.sleep(0.2)  # Adjust the sleep time as needed
                continue  # Skip to the next iteration of the loop

            # Detect cards on the table
            cards_on_table = self.find_cards_on_table()

            # Sleep before the next detection cycle
            time.sleep(3)  # Adjust the sleep time as needed

    def continuous_detection_player_cards(self):
        """Continuously detect game state changes."""

        while True:
            self.find_player_cards(1)
            time.sleep(0.5)
            # if not self.window:
            #     time.sleep(0.5)  # Adjust the sleep time as needed
            #     continue  # Skip to the next iteration of the loop
            #
            # for player_number in range(1, 7):  # Assuming 6 players in the game
            #     # player_number = 1
            #
            #     if self.check_player_card_active(player_number):
            #
            #         # print(f"Player {player_number} cards active: YES!")
            #
            #         new_cards_detected = self.find_player_cards(player_number)
            #
            #         current_cards_detected_length = len(new_cards_detected)
            #
            #         if current_cards_detected_length == 2:
            #
            #             with self.game_state_lock:
            #
            #                 current_cards_stored = self.game_state.players.get(player_number, {}).get('cards')
            #
            #                 if new_cards_detected != current_cards_stored:
            #                     # print(f"Player{player_number} NEW card = {new_cards_detected} | current_cards_stored = {current_cards_stored}")
            #
            #                     # Update the game state if there's a change
            #                     self.game_state.update_player(player_number, cards=new_cards_detected)
            #
            #                     # community_cards_count = len(self.game_state.community_cards)
            #                     # if community_cards_count < 5:
            #
            #                     # if self.game_state.hero_player_number == 0:
            #                     # self.game_state.update_player(player_number, hero=True)
            #                     # print(f"Player{player_number} is HERO!")
            #
            #                     # print(f"Player {player_number} cards updated: {current_cards_detected}")
            #
            #                 # else:
            #                 # print(f"No change in cards for Player {player_number}")
            #     # else:
            #     # print(f"Cards not active for Player {player_number}")
            #
            # # Sleep before the next detection cycle
            # time.sleep(1.0)  # Adjust the sleep time as needed

    def continuous_detection_total_pot_size(self):
        """Continuously detect game state changes."""
        while True:
            # try:
            if not self.window:
                time.sleep(0.2)
                continue

                # if self.action_processed == False:
                # Detect total pot size
            self.detect_total_pot_size()

            time.sleep(0.6)

            # except Exception as e:
            # print(f"Error in continuous detection: {e}")

    def continuous_detection_hero_action_buttons(self):
        """Continuously detect hero buttons."""
        while True:
            # try:
            if not self.window:
                time.sleep(0.2)
                continue

                # Detect hero hand combination
            # hero_card_combinations = self.detect_hero_combination_name()
            # Detect available buttons for the hero, when it's Heros turn
            self.detect_hero_buttons()

            time.sleep(0.3)

            # except Exception as e:
            # print(f"Error in continuous detection: {e}")