import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Rank 2
import torch
from lstm_model import LSTMModel
# Rank 3
from gtts import gTTS
import pygame
import os
import language_tool_python

# Initialize pygame mixer for audio
pygame.mixer.init()

# Initialize Language Tool for grammar correction
tool = language_tool_python.LanguageTool('en-US')

# Load ASL Model
model_dict_asl = pickle.load(open('./model_store/model_asl.p', 'rb'))
model_asl = model_dict_asl['model']

# Load ISL Model
model_dict_isl = pickle.load(open('./model_store/model_isl.p', 'rb'))
model_isl = model_dict_isl['model']

current_model = model_asl # Default start mode
current_mode_name = "ASL"

# Global variable for TTS language
current_language = "en"  # Default: English, can be 'te' for Telugu, 'hi' for Hindi, etc.
language_names = {"en": "English", "te": "Telugu", "hi": "Hindi"}

# --- LOAD LSTM MODEL (Rank 2 Integration) ---
device = torch.device('cpu')
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load('sign_lstm_torch.pth', map_location=device))
lstm_model.eval()  # Set to evaluation mode
lstm_labels = np.load('labels.npy')  # Load the word list
sequence_buffer = []  # This will hold the last 30 frames
last_spoken_word = ""  # Initialize the variable to track the last spoken word
# --------------------------------------------

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Text-to-Speech setup
engine = pyttsx3.init()

# label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ',
    37: '.'
}
expected_features = 42

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
last_char_time = 0  # Track when last character was registered (for auto-word-complete)
auto_complete_delay = 3.0  # Seconds of inactivity before auto-completing current word

# Detection mode: "Static" (single-frame letter detection) or "Dynamic" (LSTM word detection)
detection_mode = "Static"

# Grammar correction function using language-tool-python
def correct_sentence(text):
    """
    Corrects grammatical errors in the input text.
    
    Args:
        text (str): Raw text from sign language detection (e.g., 'Me go home')
    
    Returns:
        str: Grammatically corrected text (e.g., 'I am going home')
    """
    if not text or text.strip() == "":
        return text
    
    try:
        # Match the text against grammar rules
        matches = tool.check(text)
        # Apply corrections
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
    except Exception as e:
        print(f"Grammar correction failed: {e}")
        return text  # Return original text if correction fails

# Speak text in a separate thread
def speak_text(text):
    # Apply grammar correction before speaking
    corrected_text = correct_sentence(text)
    print(f"Original: '{text}' -> Corrected: '{corrected_text}'")
    
    def tts_thread():
        try:
            engine.say(corrected_text)
            engine.runAndWait()
        except RuntimeError:
            pass  # Ignore if TTS engine is already running

    threading.Thread(target=tts_thread, daemon=True).start()


# ============================================================================
# GUI SETUP
# ============================================================================

root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("1300x700")
root.configure(bg="#2c2f33")  # Dark theme
root.resizable(False, False)

# GUI Variables
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = StringVar(value="False")

# ============================================================================
# GUI LAYOUT
# ============================================================================

# Title (Top of window)
title_label = Label(
    root,
    text="Sign Language to Speech Conversion",
    font=("Arial", 28, "bold"),
    fg="#ffffff",
    bg="#2c2f33"
)
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Left Column: Video Feed Frame
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=640, height=480)
video_frame.grid(row=1, column=0, padx=20, pady=10, sticky="n")
video_frame.grid_propagate(False)  # Prevent frame from resizing

# Video Label (displays camera feed)
video_label = tk.Label(video_frame, bg="#000000")
video_label.pack(expand=True, fill="both")

# Button Frame (below video feed)
video_button_frame = Frame(root, bg="#2c2f33")
video_button_frame.grid(row=2, column=0, pady=(0, 20), padx=20)

# Right Column: Information Display Frame
content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, rowspan=2, sticky="n", padx=(20, 40), pady=(10, 20))

# Current Alphabet Display
Label(
    content_frame,
    text="Current Alphabet:",
    font=("Arial", 20),
    fg="#ffffff",
    bg="#2c2f33"
).pack(anchor="w", pady=(0, 10))

Label(
    content_frame,
    textvariable=current_alphabet,
    font=("Arial", 24, "bold"),
    fg="#1abc9c",
    bg="#2c2f33"
).pack(anchor="center")

# Current Word Display
Label(
    content_frame,
    text="Current Word:",
    font=("Arial", 20),
    fg="#ffffff",
    bg="#2c2f33"
).pack(anchor="w", pady=(20, 10))

Label(
    content_frame,
    textvariable=current_word,
    font=("Arial", 20),
    fg="#f39c12",
    bg="#2c2f33",
    wraplength=500,
    justify="left"
).pack(anchor="center")

# Current Sentence Display
Label(
    content_frame,
    text="Current Sentence:",
    font=("Arial", 20),
    fg="#ffffff",
    bg="#2c2f33"
).pack(anchor="w", pady=(20, 10))

Label(
    content_frame,
    textvariable=current_sentence,
    font=("Arial", 20),
    fg="#9b59b6",
    bg="#2c2f33",
    wraplength=500,
    justify="left"
).pack(anchor="center")

# ============================================================================
# GUI HELPER FUNCTIONS
# ============================================================================

def update_frame(frame):
    """
    Convert OpenCV frame (BGR) to PIL.ImageTk format and update the video label.
    
    Args:
        frame: OpenCV frame (numpy array in BGR format)
    """
    # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_rgb)
    
    # Convert to ImageTk format for Tkinter
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Update the label
    video_label.imgtk = img_tk  # Keep a reference to prevent garbage collection
    video_label.configure(image=img_tk)

def clear_sentence():
    """Clear the current word buffer and sentence."""
    global word_buffer, sentence, last_char_time
    word_buffer = ""
    sentence = ""
    last_char_time = 0
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def auto_complete_word():
    """Auto-complete the current word buffer and add it to the sentence."""
    global word_buffer, sentence, last_char_time
    if word_buffer.strip():
        speak_smart(word_buffer)
        sentence += word_buffer + " "
        current_sentence.set(sentence.strip())
        print(f"✅ Auto-completed word: '{word_buffer}' | Sentence: '{sentence.strip()}'")
        word_buffer = ""
        current_word.set("N/A")
        last_char_time = 0

def toggle_pause():
    """Toggle between pause and play states."""
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Resume")
    else:
        is_paused.set("False")
        pause_button.config(text="Pause")

def toggle_mode(event=None):
    global current_model, current_mode_name
    if current_mode_name == "ASL":
        current_model = model_isl
        current_mode_name = "ISL"
        print("✅ Switched to ISL Mode")
    else:
        current_model = model_asl
        current_mode_name = "ASL"
        print("✅ Switched to ASL Mode")

def toggle_detection_mode(event=None):
    """Toggle between Static (letter-by-letter) and Dynamic (LSTM word) detection modes."""
    global detection_mode, sequence_buffer, stabilization_buffer, last_registered_time
    if detection_mode == "Static":
        detection_mode = "Dynamic"
        sequence_buffer.clear()  # Reset LSTM buffer when switching
        stabilization_buffer.clear()  # Clear stale static data
        detection_mode_button.config(text="Mode: Dynamic")
        print("🔄 Switched to Dynamic Mode (LSTM word detection)")
    else:
        detection_mode = "Static"
        stabilization_buffer.clear()  # Clear stale buffer for fresh start
        last_registered_time = time.time()  # Reset timing so first char registers quickly
        detection_mode_button.config(text="Mode: Static")
        print("🔄 Switched to Static Mode (letter detection)")

def toggle_language(event=None):
    """Toggle TTS language between English and Telugu using the 'L' key"""
    global current_language
    if current_language == "en":
        current_language = "te"
        print("🌐 Language switched to Telugu (te)")
    else:
        current_language = "en"
        print("🌐 Language switched to English (en)")

# ============================================================================
# GUI BUTTONS (Below Video Feed)
# ============================================================================

# Clear Sentence Button
Button(
    video_button_frame,
    text="Clear Sentence",
    font=("Arial", 14, "bold"),
    command=clear_sentence,
    bg="#e74c3c",
    fg="#ffffff",
    relief="flat",
    height=2,
    width=15,
    cursor="hand2"
).grid(row=0, column=0, padx=8)

# Pause/Resume Button
pause_button = Button(
    video_button_frame,
    text="Pause",
    font=("Arial", 14, "bold"),
    command=toggle_pause,
    bg="#3498db",
    fg="#ffffff",
    relief="flat",
    height=2,
    width=12,
    cursor="hand2"
)
pause_button.grid(row=0, column=1, padx=8)

# Speak Sentence Button - speaks sentence, or word buffer if sentence is empty
def speak_current():
    text = sentence.strip() if sentence.strip() else word_buffer.strip()
    if text and text != "N/A":
        speak_smart(text)
    else:
        print("⚠️ Nothing to speak yet")

speak_button = Button(
    video_button_frame,
    text="Speak",
    font=("Arial", 14, "bold"),
    command=speak_current,
    bg="#27ae60",
    fg="#ffffff",
    relief="flat",
    height=2,
    width=15,
    cursor="hand2"
)
speak_button.grid(row=0, column=2, padx=8)

# Static/Dynamic Mode Toggle Button
detection_mode_button = Button(
    video_button_frame,
    text="Mode: Static",
    font=("Arial", 14, "bold"),
    command=toggle_detection_mode,
    bg="#8e44ad",
    fg="#ffffff",
    relief="flat",
    height=2,
    width=15,
    cursor="hand2"
)
detection_mode_button.grid(row=0, column=3, padx=8)

# Video Capture
cap = cv2.VideoCapture(0)

# Set camera feed size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

# Variables for stabilization timing
last_registered_time = time.time()
registration_delay = 1.5  # Minimum delay (in seconds) before registering the same character again

# Keyboard binding for mode switching
root.bind('s', toggle_mode)
root.bind('S', toggle_mode)

# Keyboard binding for language switching
root.bind('l', toggle_language)
root.bind('L', toggle_language)

# Keyboard binding for detection mode switching
root.bind('d', toggle_detection_mode)
root.bind('D', toggle_detection_mode)

# Rank 3
def speak_smart(text, language_code=None):
    """
    Speak text using gTTS with specified language.
    
    Args:
        text (str): Text to speak
        language_code (str): Language code ('en', 'te', 'hi', etc.). 
                            If None, uses the global current_language.
    """
    # Use global language if not specified
    if language_code is None:
        language_code = current_language
    
    # Apply grammar correction only for English
    if language_code == 'en':
        corrected_text = correct_sentence(text)
        print(f"Original: '{text}' -> Corrected: '{corrected_text}'")
    else:
        corrected_text = text
        print(f"Speaking in {language_names.get(language_code, language_code)}: '{text}'")
    
    def _speak():
        try:
            # Try Google Online TTS first (Rank 3 Feature)
            tts = gTTS(text=corrected_text, lang=language_code)
            temp_file = "temp_voice.mp3"
            tts.save(temp_file)
            
            # Load and play
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            pygame.mixer.music.unload()
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        except Exception as e:
            # Fallback to Offline TTS (Rank 1 Feature)
            print(f"gTTS failed, using pyttsx3: {e}")
            try:
                engine.say(corrected_text)
                engine.runAndWait()
            except:
                pass  # Silently fail if TTS doesn't work

    # Run in a separate thread to prevent video freezing
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, last_spoken_word, current_model, current_mode_name, last_char_time

    ret, frame = cap.read()
    if not ret:
        return

    # If paused, just update the frame and return
    if is_paused.get() == "True":
        update_frame(frame)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure valid data
            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            # === DYNAMIC MODE: LSTM word detection ===
            if detection_mode == "Dynamic":
                sequence_buffer.append(data_aux)  # Add current frame to buffer
                if len(sequence_buffer) > 30:  # Keep only last 30 frames
                    sequence_buffer.pop(0)

                if len(sequence_buffer) == 30:  # Only predict if we have 30 frames
                    # Convert to PyTorch tensor
                    input_seq = torch.tensor([sequence_buffer], dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        lstm_out = lstm_model(input_seq)
                        confidence = torch.max(lstm_out).item()
                        prediction_index = torch.argmax(lstm_out).item()
                        
                        # Filter out weak predictions - only trust confidence > 0.75
                        if confidence > 0.75: 
                            lstm_word = lstm_labels[prediction_index]
                            print(f"LSTM Detected: {lstm_word} ({confidence:.2f})")
                            
                            # Display detected word on screen (Green text for whole words)
                            cv2.putText(frame, f"WORD: {lstm_word}", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                            
                            # Speak only if it's a new word to avoid spamming
                            if lstm_word != last_spoken_word:
                                speak_smart(lstm_word)
                                last_spoken_word = lstm_word
                        else:
                            # Low confidence - don't add to sentence buffer
                            print(f"Uncertain (confidence: {confidence:.2f})") 

            # === STATIC MODE: Single-frame letter detection ===
            if detection_mode == "Static":
                # Predict gesture using current model (ASL or ISL)
                prediction = current_model.predict([np.asarray(data_aux)])
                
                # Handle both numeric and string predictions
                # ASL model returns numeric labels (as int or string like '5')
                # ISL model returns actual text labels (like 'A', 'Hello')
                pred_value = prediction[0]
                try:
                    pred_int = int(pred_value)
                    if pred_int in labels_dict:
                        predicted_character = labels_dict[pred_int]
                    else:
                        predicted_character = str(pred_value)
                except (ValueError, TypeError):
                    # ISL model returns non-numeric string labels directly
                    predicted_character = str(pred_value)

                # Stabilization logic
                stabilization_buffer.append(predicted_character)
                if len(stabilization_buffer) > 30:  # Buffer size for 1 second
                    stabilization_buffer.pop(0)

                # Count how many times the current prediction appears in the buffer
                match_count = stabilization_buffer.count(predicted_character)

                if match_count > 25:  # Stabilization threshold
                    # Register the character only if enough time has passed since the last registration
                    current_time = time.time()
                    time_since_last = current_time - last_registered_time
                    if time_since_last > registration_delay:
                        stable_char = predicted_character
                        last_registered_time = current_time  # Update last registered time
                        current_alphabet.set(stable_char)
                        print(f"📝 Static: Registered '{stable_char}' | Word: '{word_buffer + stable_char}'")
                        last_char_time = time.time()  # Track when last letter was added

                        # Handle word and sentence formation
                        if stable_char == ' ':
                            if word_buffer.strip():  # Speak word only if not empty
                                speak_smart(word_buffer)
                                sentence += word_buffer + " "
                                current_sentence.set(sentence.strip())
                                print(f"📢 Word completed: '{word_buffer}' | Sentence: '{sentence.strip()}'")
                            word_buffer = ""
                            current_word.set("N/A")
                            last_char_time = 0
                        elif stable_char == '.':
                            if word_buffer.strip():  # Speak word before adding to sentence
                                speak_smart(word_buffer)
                                sentence += word_buffer + "."
                                current_sentence.set(sentence.strip())
                                print(f"📢 Sentence ended: '{sentence.strip()}'")
                            word_buffer = ""
                            current_word.set("N/A")
                            last_char_time = 0
                        else:
                            word_buffer += stable_char
                            current_word.set(word_buffer)

            # Draw landmarks and bounding box
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # Auto-complete word after inactivity
    if detection_mode == "Static" and word_buffer.strip() and last_char_time > 0:
        if time.time() - last_char_time > auto_complete_delay:
            auto_complete_word()

    # Update the live sentence display (show sentence + word in progress)
    if sentence.strip() or word_buffer.strip():
        live_display = sentence.strip()
        if word_buffer.strip():
            live_display += (" " if live_display else "") + "[" + word_buffer + "]"
        current_sentence.set(live_display)
    else:
        current_sentence.set("N/A")

    # Draw mode, language, detection mode, and alphabet on the video feed
    cv2.putText(frame, f"MODEL: {current_mode_name} (Press 'S' to switch)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # Blue color for model
    cv2.putText(frame, f"DETECT: {detection_mode} (Press 'D' to toggle)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)  # Cyan color for detection mode
    cv2.putText(frame, f"LANGUAGE: {language_names.get(current_language, current_language)} (Press 'L' to toggle)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)  # Orange color for language
    cv2.putText(frame, f"Alphabet: {current_alphabet.get()}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color

    # Update video feed in GUI using the update_frame() function
    update_frame(frame)

    root.after(10, process_frame)


# Start processing frames
process_frame()
root.mainloop()