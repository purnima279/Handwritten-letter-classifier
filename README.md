
# 🔤 Handwritten Alphabet Recognition (A–Z)

A deep learning web application that recognizes handwritten English capital letters (A–Z) using **Python, TensorFlow/Keras, CNN, and Streamlit**.

## ✨ Features
- Recognizes all 26 uppercase letters (A–Z).
- Automatically preprocesses images into 28×28 grayscale format.
- Supports dark letters on light backgrounds and vice versa.
- Displays the predicted letter with Top-5 confidence scores.
- Interactive web interface with PNG, JPG, JPEG, and BMP upload support.

## 🧠 CNN Model Architecture

| Layer | Configuration |
|---|---|
| Input | 28×28×1 |
| Rescaling | Pixel normalization (0–1) |
| Conv2D | 32 filters, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | Feature vector conversion |
| Dense | 128 units, ReLU |
| Dropout | Overfitting prevention |
| Output | 26 units, Softmax |

## 🔬 Workflow

Upload Image → Grayscale Conversion → Crop Letter → Resize and Center → CNN Prediction → Display Results.

## 🛠️ Tech Stack
- Python 3.9–3.11
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Streamlit
- Pillow (PIL)

## 📂 Project Structure

```text
Alphabet_Project/
├── app.py
├── handwritten_alphabet_cnn_ann.keras
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation and Usage

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd handwritten-alphabet-recognition
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a handwritten letter image, and click **Predict Letter**.

## 🔮 Future Improvements
- Add a live drawing canvas.
- Extend recognition to lowercase letters and digits.
- Deploy on Streamlit Community Cloud.
- Export the model to TensorFlow Lite or ONNX.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
