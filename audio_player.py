import datetime
import pygame
import queue
import os
import threading
from pathlib import Path

class AudioPlayer:

    def __init__(self, openai_client):

        pygame.mixer.init()

        self.client             = openai_client

        self.audio_base_path    = 'Audio/'                              # Set the path to the audio files directory

        self.audio_queue        = queue.Queue()                         # Queue to hold audio files

        self.is_playing         = False                                 # Flag to check if an audio is currently being played

        self.sound_active       = True                                  # Set to False if you don't want to play any voiceovers.
        
        # Create necessary directories for audio files
        self._create_audio_directories()
        
        # Log audio initialization state
        print(f"DEBUG AUDIO: Audio player initialized, sound_active: {self.sound_active}")
    
    def _create_audio_directories(self):
        """Creates all necessary audio directories."""
        # All required directories
        directories = [
            'Audio/Player_Bet',
            'Audio/Player_Call',
            'Audio/Player_Fold',
            'Audio/Player_Is_Dealer',
            'Audio/Player_Raise',
            'Audio/Player_Check',
            'Audio/Player_Left',
            'Audio/Player_wins_pot',
            'Audio/Player_Turn',
            'Audio/Board_Stage',
            'Audio/Hero',
            'Audio/GPT_Analysis'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"DEBUG AUDIO: Created directory {directory}")
            else:
                print(f"DEBUG AUDIO: Directory {directory} already exists")
        
        # Check for missing audio files that might cause issues
        required_files = [
            'Audio/Player_Call/Player_1_Call.mp3',
            'Audio/Player_Fold/Player_1_Fold.mp3',
            'Audio/Player_Check/Player_1_Check.mp3',
            'Audio/Player_Bet/Player_1_Bet.mp3',
            'Audio/Player_Raise/Player_1_Raise.mp3'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"DEBUG AUDIO WARNING: Missing important audio files: {missing_files}")
            print(f"DEBUG AUDIO WARNING: Action sounds may not work until these files are created")


    # Audio playback queue and processing
    audio_cache = {}  # Class-level cache to avoid re-loading audio files
    
    def add_to_queue(self, file_name):
        """Add an audio file to the playback queue with optimized handling"""
        try:
            if file_name is None:
                return
                
            # Construct the full path
            full_path = self.audio_base_path + file_name
            
            # Skip verbose logging to reduce overhead
            if not os.path.exists(full_path):
                # Create directory if needed
                directory = os.path.dirname(full_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                return  # Skip if file doesn't exist
            
            # Add to queue with high priority for player actions
            self.audio_queue.put(file_name)
            
            # Start playing immediately if not already playing
            if not self.is_playing:
                # Start playback in a separate thread to avoid blocking
                threading.Thread(target=self.play_next_audio, daemon=True).start()
                
        except Exception as e:
            print(f"Audio queue error: {e}")

    def play_next_audio(self):
        """Play the next audio file with optimized non-blocking playback"""
        if self.sound_active is False or self.is_playing:
            return
            
        try:
            # Mark as playing to prevent concurrent playback
            self.is_playing = True
            
            # Process queue while there are items
            while not self.audio_queue.empty():
                file_name = self.audio_queue.get(block=False)
                full_path = self.audio_base_path + file_name
                
                # Skip missing files
                if not os.path.exists(full_path):
                    continue
                
                # Use cached audio if available
                if full_path in AudioPlayer.audio_cache:
                    sound = AudioPlayer.audio_cache[full_path]
                else:
                    # Load and cache for future use
                    try:
                        sound = pygame.mixer.Sound(full_path)
                        AudioPlayer.audio_cache[full_path] = sound
                    except Exception:
                        continue
                
                # Play with non-blocking approach
                sound.play()
                
                # Avoid the CPU-intensive waiting loop, just wait a fixed time
                # Most poker action sounds are very short
                pygame.time.wait(500)  # Wait just enough time for short sounds
                
            # Reset playing state
            self.is_playing = False
            
        except queue.Empty:
            # Queue is empty, reset state
            self.is_playing = False
        except Exception as e:
            # Handle any other errors
            self.is_playing = False
            print(f"Audio playback error: {e}")


    def convert_text_to_speech(self, text):
        
        if self.sound_active is False:
            return
        
        print("Converting text to speech...")

        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # Generate a timestamp for the file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        speech_file_name = f"GPT4_Analysis_Result_{timestamp}.mp3"

        # Path for the speech audio file with timestamp
        speech_file_path = Path(self.audio_base_path +"GPT_Analysis/"+speech_file_name)

        # Save the new speech audio to a file
        with open(speech_file_path, "wb") as f:
            f.write(response.read())
        
        #print("Speech file saved.")
        final_return_audio_path = "GPT_Analysis/"+speech_file_name

        return final_return_audio_path
    

    def play_speech(self, text):

        # Convert text to speech and get the file path
        speech_file_path = self.convert_text_to_speech(text)

        # Add the speech file to the queue
        self.add_to_queue(str(speech_file_path))


    # Player actions Audio
    def play_bet_audio(self, player_number):
        file_name = f'Player_Bet/Player_{player_number}_Bet.mp3'
        print(f"DEBUG AUDIO: Playing bet audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_call_audio(self, player_number):
        file_name = f'Player_Call/Player_{player_number}_Call.mp3'
        print(f"DEBUG AUDIO: Playing call audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_fold_audio(self, player_number):
        file_name = f'Player_Fold/Player_{player_number}_Fold.mp3'
        print(f"DEBUG AUDIO: Playing fold audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_is_dealer_audio(self, player_number):
        file_name = f'Player_Is_Dealer/Player_{player_number}_is_the_dealer.mp3'
        print(f"DEBUG AUDIO: Playing dealer audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_raise_audio(self, player_number):
        file_name = f'Player_Raise/Player_{player_number}_Raise.mp3'
        print(f"DEBUG AUDIO: Playing raise audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_check_audio(self, player_number):
        file_name = f'Player_Check/Player_{player_number}_Check.mp3'
        print(f"DEBUG AUDIO: Playing check audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_left_audio(self, player_number):
        file_name = f'Player_Left/Player_{player_number}_left.mp3'
        print(f"DEBUG AUDIO: Playing left audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_wins_the_pot_audio(self, player_number):
        file_name = f'Player_wins_pot/Player_{player_number}_wins_the_pot.mp3'
        print(f"DEBUG AUDIO: Playing wins pot audio for Player {player_number}")
        self.add_to_queue(file_name)

    def play_turn_audio(self, player_number):
        file_name = f'Player_Turn/Player_{player_number}_Turn.mp3'
        print(f"DEBUG AUDIO: Playing turn audio for Player {player_number}")
        self.add_to_queue(file_name)


    # Audio for Board Stage
    def play_board_flop_audio(self):
        self.add_to_queue('Board_Stage/Flop.mp3')

    def play_new_round_started_audio(self):
        self.add_to_queue('Board_Stage/New_round_started.mp3')

    def play_board_pre_flop_audio(self):
        self.add_to_queue('Board_Stage/Pre_flop.mp3')

    def play_board_river_audio(self):
        self.add_to_queue('Board_Stage/River.mp3')

    def play_board_turn_audio(self):
        self.add_to_queue('Board_Stage/Turn.mp3')


    # Audio for Hero Player
    def play_hero_is_big_blind_audio(self):
        self.add_to_queue('Hero/Hero_is_big_blind.mp3')

    def play_hero_is_small_blind_audio(self):
        self.add_to_queue('Hero/Hero_is_small_blind.mp3')

    def play_hero_is_the_dealer_audio(self):
        self.add_to_queue('Hero/Hero_is_the_dealer.mp3')

    def play_hero_lost_the_hand_audio(self):
        self.add_to_queue('Hero/Hero_lost_the_hand.mp3')

    def play_your_turn_audio(self):
        self.add_to_queue('Hero/Your_turn.mp3')
