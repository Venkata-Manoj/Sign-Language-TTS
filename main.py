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

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load ASL Model
model_dict_asl = pickle.load(open('./model_store/model_asl.p', 'rb'))
model_asl = model_dict_asl['model']

# Load ISL Model
model_dict_isl = pickle.load(open('./model_store/model_isl.p', 'rb'))
model_isl = model_dict_isl['model']

current_model = model_asl # Default start mode
current_mode_name = "ASL"

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

# Speak text in a separate thread
def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=tts_thread, daemon=True).start()


# GUI Setup
root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("1300x650")  # Adjusted window size for additional button
root.configure(bg="#2c2f33")  # Dark theme
root.resizable(False, False)  # Disable resizing

# Variables for GUI
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = StringVar(value="False")

# Title
title_label = Label(root, text="Sign Language to Speech Conversion", font=("Arial", 28, "bold"), fg="#ffffff", bg="#2c2f33")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)  # Reduced camera feed size
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)  # Prevent resizing

content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))  # Added right-side margin

button_frame = Frame(root, bg="#2c2f33")
button_frame.grid(row=3, column=1, pady=(10, 20),padx=(10, 20), sticky="n")  # Adjusted to fit the new button

# Video feed
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Labels
Label(content_frame, text="Current Alphabet:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(0, 10))
Label(content_frame, textvariable=current_alphabet, font=("Arial", 24, "bold"), fg="#1abc9c", bg="#2c2f33").pack(anchor="center")

Label(content_frame, text="Current Word:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), fg="#f39c12", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Current Sentence:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), fg="#9b59b6", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")  # Clear current alphabet display

def toggle_pause():
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Play")
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

# Buttons
Button(button_frame, text="Reset Sentence", font=("Arial", 16), command=reset_sentence, bg="#e74c3c", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=0, padx=10)  # Increased padding
pause_button = Button(button_frame, text="Pause", font=("Arial", 16), command=toggle_pause, bg="#3498db", fg="#ffffff", relief="flat", height=2, width=12)
pause_button.grid(row=0, column=1, padx=10)  # Consistent padding
speak_button = Button(button_frame, text="Speak Sentence", font=("Arial", 16), command=lambda: speak_text(current_sentence.get()), bg="#27ae60", fg="#ffffff", relief="flat", height=2, width=14)
speak_button.grid(row=0, column=2, padx=10)  # Added new button with proper spacing

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

# Rank 3
def speak_smart(text):
    def _speak():
        try:
            # Try Google Online TTS first (Rank 3 Feature)
            tts = gTTS(text=text, lang='en')
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
                engine.say(text)
                engine.runAndWait()
            except:
                pass  # Silently fail if TTS doesn't work

    # Run in a separate thread to prevent video freezing
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, last_spoken_word, current_model, current_mode_name

    ret, frame = cap.read()
    if not ret:
        return

    if is_paused.get() == "True":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
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

            # --- NEW LSTM LOGIC START ---
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
                    
                    # Only trust high confidence (adjust 0.8 as needed)
                    if confidence > 0.8: 
                        lstm_word = lstm_labels[prediction_index]
                        print(f"LSTM Detected: {lstm_word} ({confidence:.2f})")
                        
                        # Display detected word on screen (Green text for whole words)
                        cv2.putText(frame, f"WORD: {lstm_word}", (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        
                        # Speak only if it's a new word to avoid spamming
                        if lstm_word != last_spoken_word:
                            speak_smart(lstm_word)
                            last_spoken_word = lstm_word 
            # --- NEW LSTM LOGIC END ---

            # Predict gesture using current model (ASL or ISL)
            prediction = current_model.predict([np.asarray(data_aux)])
            
            # Handle both numeric and string predictions
            pred_value = prediction[0]
            if isinstance(pred_value, (np.str_, str)):
                # ISL model returns string labels directly
                predicted_character = str(pred_value)
            else:
                # ASL model returns numeric indices
                predicted_character = labels_dict[int(pred_value)]

            # Stabilization logic
            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 30:  # Buffer size for 1 second
                stabilization_buffer.pop(0)

            if stabilization_buffer.count(predicted_character) > 25:  # Stabilization threshold
                # Register the character only if enough time has passed since the last registration
                current_time = time.time()
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time  # Update last registered time
                    current_alphabet.set(stable_char)

                    # Handle word and sentence formation
                    if stable_char == ' ':
                        if word_buffer.strip():  # Speak word only if not empty
                            speak_text(word_buffer)
                            sentence += word_buffer + " "
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    elif stable_char == '.':
                        if word_buffer.strip():  # Speak word before adding to sentence
                            speak_text(word_buffer)
                            sentence += word_buffer + "."
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)

            # Draw landmarks and bounding box
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # Draw mode and alphabet on the video feed
    cv2.putText(frame, f"MODE: {current_mode_name} (Press 'S' to switch)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # Blue color for mode
    cv2.putText(frame, f"Alphabet: {current_alphabet.get()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color

    # Update video feed in GUI
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)


# Start processing frames
process_frame()
root.mainloop()