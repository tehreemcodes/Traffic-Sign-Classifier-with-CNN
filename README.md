# 🧠 Traffic Sign Classification using CNN (Google Colab)

This project implements a Convolutional Neural Network (CNN) to classify road signs using images, simulating a key component of self-driving car systems. The model is trained to recognize 43 categories of traffic signs such as stop signs, speed limits, yield signs, and more. The work was done using **Google Colab** and utilizes a zip-compressed dataset of labeled images.

## 📁 Dataset Structure

The dataset used is named `dataset.zip`, which contains a nested folder structure:
```
dataset/
    ├── 0/
    ├── 1/
    ├── ...
    └── 42/
```
Each subfolder (0–42) contains images representing one class of road sign in ppm format.

## 📦 Upload and Unzip Dataset

In Google Colab:
```python
from google.colab import files
uploaded = files.upload()  # Upload 'dataset.zip'

import zipfile
with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")
```

## 🔧 Preprocessing

- Images are resized to `30x30` using OpenCV.
- Normalized pixel values to the range `[0, 1]`.
- Labels are converted to one-hot encoded vectors.
- Platform-independent paths are used via `os.path.join`.

## 🧾 load_data(data_dir)

Reads and preprocesses the image data:
```python
images, labels = load_data("dataset")
```

- `images`: list of NumPy arrays (resized RGB images)
- `labels`: list of category indices (0–42)

Then:
```python
images = images / 255.0
labels = to_categorical(labels, NUM_CATEGORIES)
```

Split into train/test:
```python
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
```

## 🧠 Model Architecture

The CNN model built using TensorFlow/Keras:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])
```

Compiled with:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 📈 Training

The model is trained for 10 epochs:
```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## ✅ Evaluation

Evaluate on test set:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

## 💾 Model Saving

```python
model.save("road_sign_model.h5")
```

## 📊 Visualization

### Predictions

Plots the first 10 test images with predicted vs. true labels:
- Green title = correct prediction
- Red title = incorrect prediction

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

Displays a confusion matrix using `matplotlib` and `scikit-learn`.

## 🛠 Requirements

To run this project locally (outside Colab), you'll need:

```txt
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## 📌 Summary

- 43-class classification problem for traffic signs
- CNN built from scratch using TensorFlow/Keras
- Works on Google Colab with zip uploads
- Includes evaluation, visualization, and model export

---

**🚘 Building safer autonomous vehicles—one sign at a time.**
