
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageChops

BASE_DIR = r"D:\PYTHON\Alphabet_Project"
MODEL_PATH = os.path.join(
    BASE_DIR, "handwritten_alphabet_cnn_ann.keras"
)
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.set_page_config(
    page_title="Handwritten Alphabet Recognition",
    page_icon="🔤"
)

st.title("Handwritten Alphabet Recognition")
st.write("Upload an image containing one handwritten English letter.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

st.write("Model input shape:", model.input_shape)
st.write("Model output shape:", model.output_shape)

uploaded = st.file_uploader(
    "Choose a letter image",
    type=["png", "jpg", "jpeg", "bmp"]
)

def prepare_image(image, white_background=True):
    # Convert to grayscale.
    img = image.convert("L")

    # Automatically crop dark writing on a light background.
    # Assumes the original background is approximately white.
    inverted = ImageOps.invert(img)
    bbox = inverted.point(lambda p: 255 if p > 30 else 0).getbbox()

    if bbox:
        img = img.crop(bbox)

    # Preserve aspect ratio and fit inside 20x20.
    img.thumbnail((20, 20))

    bg = 255 if white_background else 0
    canvas = Image.new("L", (28, 28), bg)

    # Match foreground/background convention.
    if white_background:
        # Dark letter on white background.
        foreground = img
    else:
        # Bright letter on dark background.
        foreground = ImageOps.invert(img)

    canvas.paste(
        foreground,
        ((28 - img.width) // 2, (28 - img.height) // 2)
    )

    arr = np.asarray(canvas, dtype=np.float32)

    # Model's internal Rescaling layer may normalize this.
    return arr

if uploaded:
    original = Image.open(uploaded).convert("L")
    st.image(original, caption="Uploaded image", width=250)

    background = st.radio(
        "Image format to test",
        [
            "Dark letter on white background",
            "White letter on dark background"
        ]
    )

    if st.button("Predict Letter"):
        white_bg = background == "Dark letter on white background"
        arr = prepare_image(original, white_background=white_bg)

        st.image(
            arr.astype(np.uint8),
            caption="Processed 28x28 image",
            width=200
        )

        batch = arr[np.newaxis, ..., np.newaxis]
        try:
            probs = model.predict(batch, verbose=0)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        if len(probs) != 26:
            st.error(
                f"Expected 26 output classes, but model returned {len(probs)}."
            )
            st.stop()

        idx = int(np.argmax(probs))
        st.success(f"Predicted letter: {LETTERS[idx]}")
        st.write(f"Top prediction score: {probs[idx] * 100:.2f}%")

        st.subheader("Top 5 predictions")
        for i in np.argsort(probs)[-5:][::-1]:
            st.write(f"{LETTERS[i]}: {probs[i] * 100:.2f}%")
