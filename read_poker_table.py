import mss
import re
import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime
import asyncio
from asyncio import Lock
import time
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
import cProfile
import time
import random
import sys
import platform
import os

os.environ['SDL_VIDEODRIVER'] = 'x11'

import pytesseract

# Only import PaddleOCR on Windows as a fallback
if platform.system() == 'Windows':
    try:
        
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not available, falling back to pytesseract only")

from scipy.ndimage import rotate

matplotlib.use('TkAgg')


class ReadPokerTable:

    def __init__(self, poker_window, hero_info, hero_hand_range, hero_action, poker_assistant, game_state):

        self.game_state_lock = asyncio.Lock()  # Use asyncio.Lock instead of Lock

        self.hero_info = hero_info

        self.hero_hand_range = hero_hand_range

        self.hero_action = hero_action

        self.game_state = game_state

        self.poker_assistant = poker_assistant

        self.save_screenshots = False

        self.shutdown_flag = asyncio.Event()

        self.tasks = []

        self.scaling_factor = 1

        self.screenshot_lock = asyncio.Lock()  # Use asyncio.Lock instead of Lock

        self.last_action_time = {}

        self.similarities = []

        self.pending_actions = {}

        self.action_timeout = 2.0

        self.last_stack_updates = {}

        self.ocr_cache = {}

        self.ocr_cache_lock = asyncio.Lock()  # Use asyncio.Lock instead of Lock

        self.player_detection_times = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        self.last_active_player = 1

        self.last_action_player = 0

        self.hero_buttons_active = {}

        self.action_processed = False

        self.last_detected_cards = []

        self.window = poker_window

        self.poker_window_width = self.window.width

        self.poker_window_height = self.window.height

        self.poker_window_left = self.window.left

        self.poker_window_top = self.window.top

        self.window_activation_error_reported = False

        self.cards_on_table = False

        self.image_dict = {}

        self.detected_text_dict = {}

        self.photo = None

        # Set the tesseract command path based on the current OS
        if os.name == 'nt':  # Windows
            self.tesseract_cmd = r'C:\Users\Admin\Desktop\PokerGPT\tesseract\tesseract.exe'
        else:  # MacOS or Linux
            self.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

        # Configure pytesseract to use the correct path
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

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

        # Initialize OCR engine based on platform
        if platform.system() == 'Windows':
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                print("PaddleOCR initialized successfully")
            except:
                self.ocr = None
                print("Failed to initialize PaddleOCR, will use pytesseract only")
        else:
            # On Mac/Linux, we'll use pytesseract only
            self.ocr = None
            print("Using pytesseract OCR engine on Mac/Linux")

    def is_pixel_white(self, pixel, min_white=230, max_white=255):
        """
        Check if a pixel is within the white range.
        """
        r, g, b = pixel
        return all(min_white <= value <= max_white for value in (r, g, b))

    def capture_screen_area(self, relative_x, relative_y, width, height, resize_dimensions=None, filename=None,
                            whole_screen=False):
        """
        Capture a screen area based on relative coordinates for position and fixed pixel values for size.
        """
        # Calculate absolute position based on cached relative coordinates
        abs_x = int(self.poker_window_left + self.poker_window_width * relative_x)
        abs_y = int(self.poker_window_top + self.poker_window_height * relative_y)

        # Calculate actual dimensions based on window size
        actual_width = int(width * self.poker_window_width / 1920)  # Assuming 1920x1080 base resolution
        actual_height = int(height * self.poker_window_height / 1080)

        with mss.mss() as sct:
            monitor = {"top": abs_y, "left": abs_x, "width": actual_width, "height": actual_height}

            if whole_screen:
                monitor = sct.monitors[0]

            screenshot = sct.grab(monitor)
            screenshot = np.array(screenshot)

            # if self.save_screenshots or filename:
            #     filepath = f'Screenshots/{filename if filename else datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.png'
            #     mss.tools.to_png(screenshot.rgb, screenshot.size, output=filepath)

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

    async def detect_player_turn(self):
        """Loop through all players and detect which player's turn it is with optimized processing."""
        try:
            # Player turn indicator positions (relative coordinates)
            gray_background_region = {
                1: (0.565, 0.915),
                2: (0.170, 0.736),
                3: (0.199, 0.34),
                4: (0.561, 0.231),
                5: (0.918, 0.34),
                6: (0.951, 0.736)
            }

            # Process each player position - check the white highlight
            for player_number, (region_x, region_y) in gray_background_region.items():
                try:
                    # Fast capture with minimal processing
                    pixel_color = self.capture_screen_area(region_x, region_y, 10, 10)

                    if pixel_color is None:
                        continue

                    # Fast color check - just check a few pixels
                    pixel_array = np.array(pixel_color)
                    if pixel_array.size == 0:  # Check if array is empty
                        continue

                    pixel_value = pixel_array[0][0][:3]  # Get just one representative pixel

                    if self.is_pixel_white(pixel_value):
                        # White pixel found - this player's turn
                        async with self.game_state_lock:
                            if self.game_state.get_current_player_turn() != player_number:
                                # Handle player turn change
                                self.game_state.update_player(player_number, turn=True)
                                self.last_active_player = player_number

                                # Start hero analysis in a separate task if it's hero's turn
                                if player_number == self.game_state.hero_player_number:
                                    asyncio.create_task(
                                        self.game_state.analyze_hand(self.poker_assistant.openai_client)
                                    )

                    return player_number
                except (IndexError, TypeError, ValueError) as e:
                    # Log the specific error for this player position
                    error_log_dir = "error_logs/player_turn"
                    os.makedirs(error_log_dir, exist_ok=True)
                    error_log_file = os.path.join(error_log_dir,
                                                  f"player{player_number}_error_{datetime.now().strftime('%Y%m%d')}.txt")

                    with open(error_log_file, "a") as f:
                        f.write(
                            f"{time.strftime('%Y%m%d_%H%M%S_%f')} - Error checking player {player_number} turn: {str(e)}\n")

                    continue  # Skip to the next player

            return self.last_active_player

        except Exception as e:
            # Detailed logging of the error
            error_log_dir = "error_logs/player_turn"
            os.makedirs(error_log_dir, exist_ok=True)
            error_log_file = os.path.join(error_log_dir, f"detect_turn_error_{datetime.now().strftime('%Y%m%d')}.txt")

            with open(error_log_file, "a") as f:
                f.write(f"{time.strftime('%Y%m%d_%H%M%S_%f')} - Error in detect_player_turn: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())

            # Return the last known active player as a fallback
            return self.last_active_player

    async def detect_total_pot_size(self):
        """Detect the total pot size on the table."""
        detected_text = self.detect_text(0.445, 0.312, 110, 28)

        if detected_text is None:
            return

        try:
            cleaned_text = detected_text.replace(',', '').replace('Pot:', '').replace('$', '').replace('.', '').strip()
            new_pot_size = float(cleaned_text) / 100

            async with self.game_state_lock:
                if new_pot_size != self.game_state.total_pot:
                    self.game_state.update_total_pot(new_pot_size)

        except (ValueError, Exception) as e:
            pass

        return detected_text

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

    # Cache for player turn detection to avoid unnecessary processing
    player_turn_cache = {'timestamp': 0, 'player': None}

    async def detect_player_stack_and_action(self, player_number):
        """Detect and process player actions with ultra-fast detection."""
        # Define regions for all players with updated coordinates
        player_regions = {
            1: {'stack': (0.444, 0.912), 'action': (0.461, 0.840)},
            2: {'stack': (0.058, 0.741), 'action': (0.074, 0.675)},
            3: {'stack': (0.089, 0.340), 'action': (0.105, 0.274)},
            4: {'stack': (0.447, 0.241), 'action': (0.464, 0.175)},
            5: {'stack': (0.808, 0.338), 'action': (0.824, 0.273)},
            6: {'stack': (0.840, 0.741), 'action': (0.855, 0.675)}
        }

        # Use doubled dimensions since we're not resizing anymore
        action_width = 158  
        action_height = 26  
        stack_width = 200   
        stack_height = 30   
        
        region_action_x, region_action_y = player_regions[player_number]['action']
        region_stack_x, region_stack_y = player_regions[player_number]['stack']
        
        # Action detection first (speed priority)
        try:
            # Capture action area
            action_img = self.capture_screen_area(region_action_x, region_action_y, action_width, action_height)
            
            if action_img is not None:
                # Convert to grayscale
                action_array = np.array(action_img)
                action_gray = cv2.cvtColor(action_array, cv2.COLOR_BGR2GRAY)
                
                # Use three different OCR configs to maximize chance of detection
                action_configs = [
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist="BCFRALINSadeghiklnorst-0123456789$."',
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist="BCFRALINSadeghiklnorst-0123456789$."',
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist="BCFRALINSadeghiklnorst-0123456789$."'
                ]
                
                detected_action = None
                amount_in_text = None
                detected_text = None
                
                # Try each OCR config until we find something
                for config in action_configs:
                    detected_text = pytesseract.image_to_string(action_gray, config=config).lower().strip()
                    if detected_text:
                        # Extract any numbers from the text
                        all_numbers = re.findall(r'\d+\.?\d*', detected_text)
                        if all_numbers:
                            try:
                                amount_in_text = float(max(all_numbers, key=float))
                            except:
                                pass
                        
                        # Simple substring matching for poker actions
                        if "call" in detected_text:
                            detected_action = 'call'
                        elif "fold" in detected_text:
                            detected_action = 'fold'
                        elif "raise" in detected_text:
                            detected_action = 'raise'
                        elif "check" in detected_text:
                            detected_action = 'check'
                        elif "bet" in detected_text:
                            detected_action = 'bet'
                        elif "all-in" in detected_text or "all in" in detected_text:
                            detected_action = 'all-in'
                            
                        if detected_action:
                            break  # Found an action, no need to try other configs
                
                # If we found an action, try to get stack info for bet amount calculation
                if detected_action:
                    # Calculate bet amount
                    bet_amount = None
                    
                    # Get stack info in parallel while processing action
                    prev_stack = None
                    current_stack = None
                    
                    if detected_action in ['raise', 'bet', 'call', 'all-in']:
                        # Try to calculate stack first
                        try:
                            stack_image = self.capture_screen_area(region_stack_x, region_stack_y, stack_width, stack_height)
                            if stack_image is not None:
                                stack_array = np.array(stack_image)
                                stack_gray = cv2.cvtColor(stack_array, cv2.COLOR_BGR2GRAY)
                                
                                # Get stack text
                                stack_text = pytesseract.image_to_string(
                                    stack_gray, 
                                    config=r'--oem 3 --psm 7 -c tessedit_char_whitelist="0123456789.$,"'
                                ).strip()
                                
                                if stack_text:
                                    # Clean up and convert to float
                                    cleaned_text = stack_text.replace(',', '').replace('$', '').strip()
                                    
                                    try:
                                        current_stack = float(cleaned_text)
                                        prev_stack = self.game_state.players.get(player_number, {}).get('stack_size', None)
                                        
                                        if prev_stack is not None and current_stack != prev_stack:
                                            # Update game state silently
                                            async with self.game_state_lock:
                                                self.game_state.update_player(player_number, stack_size=current_stack)
                                                
                                        # Calculate bet amount from stack difference
                                        if prev_stack is not None and current_stack < prev_stack:
                                            bet_amount = prev_stack - current_stack
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Output action with amount (priority order)
                    if detected_action in ['raise', 'bet', 'call', 'all-in']:
                        # First try to get amount from the OCR text
                        if amount_in_text is not None:
                            bet_amount = amount_in_text
                            print(f"\n>> PLAYER {player_number}: {detected_action.upper()} ${bet_amount:.2f}")
                        # Then from stack difference
                        elif bet_amount is not None:
                            print(f"\n>> PLAYER {player_number}: {detected_action.upper()} ${bet_amount:.2f}")
                        # Finally just the action if no amount found
                        else:
                            print(f"\n>> PLAYER {player_number}: {detected_action.upper()}")
                    else:
                        # For check/fold actions
                        print(f"\n>> PLAYER {player_number}: {detected_action.upper()}")
                    
                    # Update game state
                    async with self.game_state_lock:
                        if detected_action in ['fold', 'check']:
                            self.game_state.update_player(player_number, action=detected_action.capitalize())
                            self.last_action_player = player_number
                        elif detected_action in ['call', 'raise', 'bet', 'all-in'] and bet_amount is not None:
                            self.game_state.update_player(
                                player_number,
                                amount=bet_amount,
                                action=detected_action.capitalize(),
                                stack_size=current_stack
                            )
                            # Store this stack update with timestamp
                            self.last_stack_updates[player_number] = {
                                'change': bet_amount,
                                'time': time.time()
                            }
                            self.last_action_player = player_number
                        elif detected_action in ['call', 'raise', 'bet', 'all-in']:
                            # Store as pending if we couldn't calculate amount
                            self.pending_actions[player_number] = {
                                'action': detected_action,
                                'time': time.time()
                            }
                    return  # Exit after handling action
                    
        except Exception:
            pass

    async def check_all_players_sequentially(self):
        """Check all players with true parallel processing and maximum speed."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    continue  # No sleep at all to process as fast as possible

                # Create tasks for all players simultaneously
                detection_tasks = [
                    self.detect_player_stack_and_action(player_number)
                    for player_number in range(1, 7)
                ]

                # Run all player checks in parallel
                await asyncio.gather(*detection_tasks)

                # No sleep at all - process as fast as possible
                
            except Exception:
                pass  # No sleep on exception either

    async def continuous_detection_player_turn(self):
        """Continuously detect game state changes."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    await asyncio.sleep(0.2)
                    continue

                # Detect active players turn
                await self.detect_player_turn()

                # Process all players at a consistent rate
                await asyncio.sleep(0.1)  # Short sleep between checks to prevent CPU overload

            except Exception as e:
                print(f"Error in continuous detection player turn: {e}")
                await asyncio.sleep(0.2)

    async def continuous_detection_dealer_button(self):
        """Continuously detect dealer button position."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    await asyncio.sleep(0.2)
                    continue

                dealer_pos = await self.find_dealer_button(self.dealer_button_image)
                
                # Check for dealer every 3 seconds
                await asyncio.sleep(3)

            except Exception as e:
                await asyncio.sleep(0.2)

    async def continuous_detection_table_cards(self):
        """Continuously detect cards on the table."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    await asyncio.sleep(0.2)
                    continue

                # Detect cards on the table
                board_cards = await self.find_cards_on_table()
                
                # Check board cards every 3 seconds
                await asyncio.sleep(3)

            except Exception as e:
                await asyncio.sleep(0.2)

    async def continuous_detection_hero_cards(self):
        """Continuously detect hero cards."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    await asyncio.sleep(0.2)
                    continue

                await self.find_hero_cards()
                
                # Check hero cards every 3 seconds
                await asyncio.sleep(3)

            except Exception as e:
                await asyncio.sleep(0.2)

    async def detect_hero_buttons(self):
        """Check the three buttons for the presence of the specified white color and detect text."""
        button_width = 120
        button_height = 50

        button_positions = [
            (0.516, 0.907),
            (0.679, 0.907),
            (0.842, 0.907),
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

                    if "Raise" in cleaned_button_text:
                        self.hero_buttons_active[i] = {"action": "Raise", "pos": (x, y)}
                        any_button_active = True
                    elif "Bet" in cleaned_button_text:
                        self.hero_buttons_active[i] = {"action": "Bet", "pos": (x, y)}
                        any_button_active = True
                    elif any(keyword in cleaned_button_text for keyword in ["Fold", "Call", "Check", "Resume", "Cash"]):
                        any_button_active = True
                        self.hero_buttons_active[i] = {"action": cleaned_button_text, "pos": (x, y)}

                    await asyncio.sleep(0.4)

        if any_button_active:
            if not self.action_processed:
                if self.game_state.round_count > 0:
                    if self.game_state.current_board_stage == 'Pre-Flop':
                        hero_role = self.game_state.players[self.game_state.hero_player_number].get('role')
                        hero_cards = self.game_state.players[self.game_state.hero_player_number].get('cards')

                        print(f"HERO CARDS: {hero_cards}")

                        is_playable_card = False
                        if hero_cards:
                            is_playable_card = self.hero_hand_range.is_hand_in_range(hero_cards)

                        if is_playable_card:
                            analysis_thread = asyncio.create_task(self.analyze_and_log())
                            print(f"PLAYABLE CARD: {hero_cards} in {hero_role} ROLE")
                        else:
                            await self.hero_action.execute_action(None, "Fold", None)
                            self.game_state.update_player(self.game_state.hero_player_number, action='Fold')

                            self.hero_info.update_action_count(
                                self.game_state.round_count,
                                self.game_state.players[self.game_state.hero_player_number].get('role'),
                                self.game_state.current_board_stage,
                                'Fold'
                            )

                            print(f"UNPLAYABLE CARD: {hero_cards} in {hero_role} ROLE")
                    else:
                        analysis_thread = asyncio.create_task(self.analyze_and_log())

                self.action_processed = True
        else:
            self.hero_buttons_active = {}
            self.action_processed = False

    async def analyze_and_log(self):

        action_result = await self.poker_assistant.AnalyzeAI(self.hero_buttons_active, self.game_state.get_ai_log())

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

    async def find_dealer_button(self, button_template):
        """Find the dealer button using SSIM comparison with a reference image."""
        dealer_button_regions = {
            1: (0.425, 0.690),
            2: (0.212, 0.662),
            3: (0.214, 0.339),
            4: (0.445, 0.283),
            5: (0.764, 0.338),
            6: (0.765, 0.663)
        }
        
        # Dealer button dimensions - using the exact dimensions of the reference image
        width, height = 47, 34
        
        # Load the reference dealer button image
        reference_button = cv2.imread('images/dealer_button.png', cv2.IMREAD_GRAYSCALE)
        if reference_button is None:
            print("Warning: Could not load dealer button reference image.")
            return None
            
        # Resize reference to exact dimensions if needed
        reference_button = cv2.resize(reference_button, (width, height))
            
        highest_similarity = 0
        best_match_position = None
        
        for player_number, (region_x, region_y) in dealer_button_regions.items():
            screenshot = self.capture_screen_area(region_x, region_y, width, height)
            if screenshot is None:
                continue
                
            # Convert to grayscale for comparison
            screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
            
            # Resize to match reference dimensions
            screenshot_gray = cv2.resize(screenshot_gray, (width, height))
            
            # Calculate SSIM between the reference and the screenshot
            try:
                similarity = ssim(reference_button, screenshot_gray)
                
                # Store if it's the best match so far
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_position = player_number
            except Exception as e:
                print(f"Error calculating similarity for position {player_number}: {e}")
        
        # If we found a good match (similarity > threshold)
        if highest_similarity > 0.6 and best_match_position is not None:
            async with self.game_state_lock:
                if self.game_state.dealer_position != best_match_position:
                    self.game_state.update_dealer_position(best_match_position)
                    self.game_state.dealer_position = best_match_position
                    # Print dealer position when it changes
                    print(f"Dealer: Player {best_match_position}")

                    if self.game_state.round_count > 1:
                        if self.game_state.round_count % 12 == 0:
                            asyncio.create_task(
                                self.poker_assistant.analyze_players_gpt4(self.game_state.all_round_logs)
                            )

                    self.game_state.reset_for_new_round()

            return best_match_position

        return None

    async def find_cards_on_table(self):
        """Find cards on the table - simplified version."""
        # Define card regions on the table
        card_regions = {
            "flop1": (0.352, 0.431),
            "flop2": (0.426, 0.431),
            "flop3": (0.499, 0.431),
            "turn": (0.572, 0.431),
            "river": (0.644, 0.431)
        }

        detected_cards = []

        for position, (x, y) in card_regions.items():
            # Check if there's a card at this position
            card_width = 55
            card_height = 65
            card_image = self.capture_screen_area(x, y, card_width, card_height)

            if card_image is not None:
                # Convert to grayscale for better text detection
                card_array = np.array(card_image)
                if len(card_array.shape) < 3 or card_array.size == 0:
                    continue

                gray_card = cv2.cvtColor(card_array, cv2.COLOR_BGR2GRAY)

                # Use OCR to detect card value and suit
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="23456789TJQKA♣♦♥♠"'
                card_text = pytesseract.image_to_string(gray_card, config=custom_config).strip()

                # Parse card text (simplistic approach)
                if card_text and len(card_text) >= 1:
                    # Clean up the detected text
                    card_text = card_text.replace("\n", "").strip()

                    # If we detect a card, add it to our list
                    if card_text:
                        detected_cards.append(card_text)

        # Update game state if cards found
        if detected_cards:
            async with self.game_state_lock:
                self.game_state.update_table_cards(detected_cards)

        return detected_cards

    async def find_hero_cards(self):
        """Find the hero's cards."""
        # Define regions for hero cards
        hero_card_regions = {
            "card1": (0.382, 0.73),
            "card2": (0.427, 0.73)
        }

        hero_cards = []

        for position, (x, y) in hero_card_regions.items():
            # Check for a card at this position
            card_width = 55
            card_height = 65
            card_image = self.capture_screen_area(x, y, card_width, card_height)

            if card_image is not None:
                # Convert to grayscale for OCR
                card_array = np.array(card_image)
                if len(card_array.shape) < 3 or card_array.size == 0:
                    continue

                gray_card = cv2.cvtColor(card_array, cv2.COLOR_BGR2GRAY)

                # Use OCR to detect card value and suit
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="23456789TJQKA♣♦♥♠"'
                card_text = pytesseract.image_to_string(gray_card, config=custom_config).strip()

                # Parse card text
                if card_text and len(card_text) >= 1:
                    # Clean up the detected text
                    card_text = card_text.replace("\n", "").strip()

                    # If we detect a card, add it to our list
                    if card_text:
                        hero_cards.append(card_text)

        # Update game state if hero cards have changed
        if hero_cards and len(hero_cards) == 2:
            hero_player = self.game_state.hero_player_number
            current_cards = self.game_state.players.get(hero_player, {}).get('cards', [])

            if hero_cards != current_cards:
                async with self.game_state_lock:
                    self.game_state.update_player(hero_player, cards=hero_cards)
                # Print hero cards to console
                print(f"Hero cards: {' '.join(hero_cards)}")

        return hero_cards

    async def monitor_player_stacks(self):
        """Monitor player stacks silently."""
        while not self.shutdown_flag.is_set():
            try:
                if not self.window:
                    await asyncio.sleep(1)
                    continue
                
                # Much less frequent checks to reduce console spam
                await asyncio.sleep(5)
                
            except Exception as e:
                await asyncio.sleep(1)
                
    async def start_detection(self):
        """Start all detection tasks."""
        try:            
            # Start regular detection tasks
            self.tasks = [
                asyncio.create_task(self.check_all_players_sequentially()),
                asyncio.create_task(self.continuous_detection_dealer_button()),
                asyncio.create_task(self.continuous_detection_player_turn()),
                asyncio.create_task(self.continuous_detection_table_cards()),
                asyncio.create_task(self.continuous_detection_hero_cards()),
                asyncio.create_task(self.monitor_player_stacks()),
            ]

            print("Started all detection tasks")
            
            # Return the tasks so they can be tracked
            return self.tasks
            
        except Exception as e:
            print(f"Error starting detection: {e}")
            return []