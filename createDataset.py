import os
import pickle
import cv2
import mediapipe as mp
import warnings

from tqdm import tqdm  # 🔹 ADDED: progress bar

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

DATA_DIR = './Data_ISL'
EXPECTED_FEATURES = 42  # 21 landmarks × (x, y)

data = []
labels = []

# -----------------------------
# Count total images (for tqdm)
# -----------------------------
total_images = sum(
    len(os.listdir(os.path.join(DATA_DIR, d)))
    for d in os.listdir(DATA_DIR)
)

# -----------------------------
# Main processing loop
# -----------------------------
with tqdm(total=total_images, desc="Processing images") as pbar:  # 🔹 ADDED
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)

        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            # Read image
            img = cv2.imread(os.path.join(dir_path, img_path))

            if img is None:  # 🔹 SAFETY CHECK
                pbar.update(1)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image with Mediapipe Hands
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect x and y values
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    # Normalize landmarks
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                # Check feature size before saving
                if len(data_aux) == EXPECTED_FEATURES:
                    data.append(data_aux)
                    labels.append(dir_)

            pbar.update(1)  # 🔹 UPDATE progress bar after each image

# -----------------------------
# Save dataset
# -----------------------------
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset saved successfully!")
print(f"📦 Total samples: {len(data)}")
