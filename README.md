# 🇰🇪 Kenyan Sign Language Recognition (KSL) - Open Research Project

This project is an open-source effort to build a real-time Sign Language Recognition system focused on **Kenyan Sign Language (KSL)**. It's designed to be an accessible base for researchers, students, or developers who wish to expand or adapt the model further.

There are **two main code paths** included in this project:

- **🧪 Tutorial Code:**  
  Used during data collection and debugging. This version helps with **retaking actions** that didn’t perform well and is useful for experimentation or when dealing with noisy/incorrect samples.

- **⚙️ Pure Final Code:**  
  A clean, production-ready version with minimal clutter. Use this if you want a **stable baseline** to improve, modify, or integrate into other systems.


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

---

## 🧠 Model Training

### Architecture

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
```

Training Setup
python
Copy
Edit
```
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop])
```
🗣️ Real-Time Prediction & Sentence Construction
Logic for Sentence Construction


python
Copy
Edit
```
def build_sentence(words):
    word_set = set(words)
    if word_set >= {'help-me', 'please', 'me'}:
        return "Please help me."
    elif word_set >= {'who', 'you'}:
        return "Who are you?"
    elif word_set >= {'me', 'name'}:
        return "My name is ..."
    elif word_set >= {'how', 'you'}:
        return "How are you?"
    elif word_set >= {'where', 'home'}:
        return "Where is home?"
    else:
        return " ".join(words).capitalize() + "."
```
## 💡 How to Extend the Project

- **Add More Actions:**  
  Add new gestures to the `actions` list and collect additional data using the data collection pipeline.

- **Improve Sentence Logic:**  
  Expand the `build_sentence()` function with rule-based logic or integrate natural language processing (NLP) for dynamic sentence generation.

- **Add Voice Output:**  
  Integrate libraries like [`pyttsx3`](https://pypi.org/project/pyttsx3/) or [`gTTS`](https://pypi.org/project/gTTS/) to convert recognized sign sequences into speech.

- **Improve the Model:**  
  Gather more diverse training data, fine-tune hyperparameters, increase model complexity, or try different architectures for better accuracy.

---

## 👨🏽‍🔬 For Researchers

This project is designed for **open collaboration**. If you're a researcher or developer interested in African sign languages, you are encouraged to:

- Expand the **action set** for Kenyan Sign Language (KSL)
- Improve the **sentence construction logic**
- Study **regional dialects** and variations in KSL
- Add **multilingual translation support**

Please consider contributing back to help improve **accessibility** and support **the Kenyan Deaf community**.

---

## 🤝 Contributing

1. **Fork** this repository  
2. **Clone** your fork locally  
3. **Make changes** and commit  
4. **Push** to your fork  
5. **Create a pull request**

We welcome pull requests for improving code quality, adding new features, or expanding the sign language dataset.

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with proper attribution.

---

## 👋 Final Note

Kenyan Sign Language deserves broader recognition and technological support.

This project is only the beginning.  
We invite developers, linguists, and researchers to help build a smarter, inclusive digital future for the Deaf community.

**– Ibrahim Shedoh(Project Author, Nairobi, Kenya)**


