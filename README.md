# 🇰🇪 Kenyan Sign Language Recognition (KSL) - Open Research Project

This project is an open-source effort to build a real-time Sign Language Recognition system focused on **Kenyan Sign Language (KSL)**. It's designed to be an accessible base for researchers, students, or developers who wish to expand or adapt the model further.

---

## 📌 Project Goals

- Collect and process KSL signs using webcam and MediaPipe Holistic.
- Train an LSTM deep learning model to recognize gestures in real time.
- Construct **rule-based natural language sentences** from detected signs.
- Open the project to encourage **collaboration and extension** of the KSL dataset.

---

## 📁 Project Structure

├── MP_Data/ # Collected and saved keypoint data for each action
│ └── [action]/[sequence]/[frame].npy
├── model/ # Trained model file
│ └── ksl_model.h5
├── main.py # Main real-time detection and sentence construction script
├── utils.py # Helper functions: landmark drawing, keypoint extraction
├── README.md # This file

yaml
Copy
Edit

---

## 🧪 Actions (Signs)

Currently, the model is trained on **30 signs**, including:

['me', 'you', 'friend', 'name', 'mine', 'who', 'how', 'please', 'help-me', 'wait',
'now', 'home', 'where', 'give-me', 'thank-you', 'polite', 'hello', 'good',
'mother', 'father', 'uncle', 'cousing', 'brother', 'sister', 'doughter',
'parent', 'relative', 'yes', 'no', 'sorry']

yaml
Copy
Edit

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – for video capture and rendering
- **MediaPipe Holistic** – for hand, pose, and face landmark detection
- **TensorFlow / Keras** – for building and training the LSTM model
- **NumPy** – for data handling
- **Matplotlib** – for visualization (optional)

---

## 📥 Data Collection

Each action is recorded as:

- `30 sequences` (videos)
- Each containing `30 frames`
- Each frame is stored as `.npy` arrays of extracted keypoints

Example snippet:

1. Data Collection Snippet
python
Copy
Edit
# === OPEN: Data Collection ===
cap = cv2.VideoCapture(0)
for action in actions_to_collect:
    for sequence in range(30):
        for frame_num in range(30):
            ...
            keypoints = extract_keypoints(results)
            np.save(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), keypoints)
# === CLOSE: Data Collection ===
2. Model Architecture
python
Copy
Edit
# === OPEN: Model Architecture ===
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
# === CLOSE: Model Architecture ===
3. Model Training
python
Copy
Edit
# === OPEN: Model Training ===
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop])
# === CLOSE: Model Training ===
4. Real-Time Prediction
python
Copy
Edit
# === OPEN: Real-Time Prediction ===
sequence = []
sentence = []
predictions = []

if len(sequence) == 30:
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    predicted_class = np.argmax(res)
    ...
# === CLOSE: Real-Time Prediction ===
5. Rule-Based Sentence Construction
python
Copy
Edit
# === OPEN: Rule-Based Sentence Construction ===
def build_sentence(words):
    word_set = set(words)
    if word_set >= {'help-me', 'please', 'me'}:
        return "Please help me."
    elif word_set >= {'who', 'you'}:
        return "Who are you?"
    elif word_set >= {'me', 'name'}:
        return "My name is ..."
    ...
    return " ".join(words).capitalize() + "."
# === CLOSE: Rule-Based Sentence Construction ===

💡 How to Extend the Project
Add More Actions
Extend the actions list and collect new gesture data using the existing data collection pipeline.

Improve Sentence Logic
Enhance the build_sentence() function or integrate NLP tools to allow smarter sentence generation based on detected signs.

Add Voice Output
Use libraries like pyttsx3 (offline) or gTTS (online) to speak out the predicted sentences.

Improve Model Accuracy
Gather more diverse training data, fine-tune the model architecture, or add advanced layers like Dropout and BatchNormalization.

👨🏽‍🔬 For Researchers
This project is built for open collaboration. You’re encouraged to:

Expand the action set for Kenyan Sign Language

Create more expressive and accurate sentence logic

Study and map regional sign variations

Add multilingual translation capabilities

Your contributions can advance digital inclusion in Kenya and worldwide.

🤝 Contributing
Fork this repository

Clone your fork:

bash
Copy
Edit
git clone https://github.com/your-username/kenyan-sign-language-recognition.git
Make your changes and commit

Push to your fork

Open a pull request

📄 License
This project is licensed under the MIT License. See LICENSE for details.

👋 Final Note
Kenyan Sign Language deserves greater visibility and technical support. This project lays the groundwork for that effort — and you are warmly invited to help improve it.

– Ibrahim Shedoh
Nairobi, Kenya
