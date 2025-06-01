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

```python
cap = cv2.VideoCapture(0)
for action in actions_to_collect:
    for sequence in range(30):
        for frame_num in range(30):
            ...
            keypoints = extract_keypoints(results)
            np.save(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), keypoints)
🧠 Model Training
The LSTM model is trained using the extracted keypoints.

Architecture
python
Copy
Edit
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
Training Setup
python
Copy
Edit
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop])
🎯 Real-Time Prediction & Sentence Construction
Real-Time Detection
python
Copy
Edit
sequence = []
sentence = []
predictions = []

if len(sequence) == 30:
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    predicted_class = np.argmax(res)
    ...
Rule-Based Sentence Construction
python
Copy
Edit
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
💡 How to Extend the Project
Add More Actions: Add to the actions list and collect new data.

Improve Sentence Logic: Expand build_sentence() or apply NLP.

Add Voice Output: Use pyttsx3 or gTTS for text-to-speech.

Improve Model: Collect more data, tune the model, or add layers.

👨🏽‍🔬 For Researchers
This project is designed for open contribution. You’re welcome to:

Expand the action set for Kenyan Sign Language

Build more expressive sentence logic

Analyze regional variations of signs

Add multilingual translation support

Please consider contributing back to benefit the Kenyan community and global accessibility.

🤝 Contributing
Fork this repository

Clone your fork

Make changes and commit

Push to your fork

Create a pull request

📄 License
This project is licensed under the MIT License.

👋 Final Note
Kenyan Sign Language deserves wider visibility and digital support. This project is a foundation for building that future, and you’re invited to improve it.


– Ibrahim Shedoh(Project Author, Nairobi, Kenya)
