import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import altair as alt
import json

# -------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit command)
# -------------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognizer – Advanced",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------------
# THEME / BASIC CSS
# -------------------------------------------------------
def set_theme_css(mode: str):
    if mode == "Dark":
        bg = "#0e1117"
        text = "#F8F9FA"
        card_bg = "#181C25"
    else:
        bg = "#F5F5F5"
        text = "#111111"
        card_bg = "#FFFFFF"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg};
            color: {text};
        }}
        .card {{
            background-color: {card_bg};
            padding: 1rem;
            border-radius: 0.8rem;
            margin-bottom: 1rem;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------------
@st.cache_resource
def load_mnist_model():
    # try new keras format first
    for path in ["mnist_cnn_v2.keras", "mnist_cnn.keras", "mnist_cnn.h5"]:
        try:
            model = load_model(path)
            print(f"Loaded model from {path}")
            return model
        except Exception:
            continue
    raise FileNotFoundError(
        "No model file found. Please keep 'mnist_cnn_v2.keras' or 'mnist_cnn.keras' or 'mnist_cnn.h5' "
        "in the same folder as this script."
    )


model = load_mnist_model()


# -------------------------------------------------------
# PREPROCESSING FOR PREDICTION
# -------------------------------------------------------
def preprocess_image(raw_img: Image.Image):
    """
    Convert canvas/upload image to 28x28 grayscale tensor appropriate for MNIST CNN.
    Also return a preview image to show what the model actually sees.
    """

    # 1. Convert to grayscale
    img = raw_img.convert("L")  # L = 8-bit pixels, black and white

    # 2. Numpy array [0..255]
    arr = np.array(img).astype("float32")

    # 3. Make sure background is dark and digit is bright
    # If background looks bright -> invert
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 4. Normalize to [0,1]
    arr /= 255.0

    # 5. Find bounding box of the digit and crop
    mask = arr > 0.1  # threshold
    if mask.any():
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    # 6. Make it square by padding
    h, w = arr.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype="float32")
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = arr

    # 7. Convert back to PIL for resize with antialias + slight blur
    pil = Image.fromarray((padded * 255).astype("uint8"))
    pil = pil.resize((28, 28), Image.LANCZOS)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))

    # 8. Final tensor
    final_arr = np.array(pil).astype("float32") / 255.0
    final_arr = np.expand_dims(final_arr, axis=-1)   # (28, 28, 1)
    final_arr = np.expand_dims(final_arr, axis=0)    # (1, 28, 28, 1)

    return final_arr, pil


def predict_digit(source_img: Image.Image):
    x, prep_preview = preprocess_image(source_img)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return digit, confidence, probs, prep_preview


# -------------------------------------------------------
# PROBABILITY BAR CHART (ALTAIR)
# -------------------------------------------------------
def prob_bar_chart(probs):
    probs = probs.tolist()  # ensure Python floats
    df = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability": probs
    })

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Digit:N", title="Digit"),
            y=alt.Y("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("Digit:N"),
                alt.Tooltip("Probability:Q", format=".3f")
            ],
        )
        .properties(height=260)
    )
    return chart


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
set_theme_css(theme_mode)

input_mode = st.sidebar.radio("Input Mode", ["Draw Digit", "Upload Image"])
stroke_width = st.sidebar.slider("Brush Size (Draw Mode)", 8, 40, 20)
realtime = st.sidebar.checkbox("Predict in real-time (Draw Mode)", True)

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0


# -------------------------------------------------------
# PAGE TITLE
# -------------------------------------------------------
st.title("🧠 MNIST Handwritten Digit Recognizer – Advanced")
st.write("Draw a digit **or upload an image**, and let the CNN model predict it.")


# -------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------
col_left, col_right = st.columns([1, 1])

pred_digit = None
pred_conf = None
pred_probs = None
source_image = None
preprocessed_preview = None

# --------------------- LEFT: INPUT ----------------------
with col_left:
    st.subheader("✏️ Input")

    if input_mode == "Draw Digit":
        if st.button("🧹 Clear Canvas"):
            st.session_state["canvas_key"] += 1

        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        if canvas.image_data is not None:
            source_image = Image.fromarray(canvas.image_data.astype("uint8"))

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🖼 Your Drawing")
            st.image(source_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            do_predict = realtime or st.button("🔮 Predict Digit")

            if do_predict:
                pred_digit, pred_conf, pred_probs, preprocessed_preview = predict_digit(source_image)

    else:  # Upload Image
        uploaded = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            source_image = Image.open(uploaded)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🖼 Uploaded Image")
            st.image(source_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔮 Predict Digit"):
                pred_digit, pred_conf, pred_probs, preprocessed_preview = predict_digit(source_image)


# ------------------- RIGHT: RESULT ----------------------
with col_right:
    st.subheader("📊 Prediction & Analysis")

    if pred_digit is not None:
        # Prediction card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Prediction")
        st.write(f"**Predicted Digit:** `{pred_digit}`")
        st.write(f"**Confidence:** `{pred_conf:.4f}`")
        st.markdown("</div>", unsafe_allow_html=True)

        # Confidence gauge
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📏 Confidence Gauge")
        st.progress(min(max(pred_conf, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        # Preprocessed preview (what model sees)
        if preprocessed_preview is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("👀 Model Input Preview")
            st.image(preprocessed_preview.resize((140, 140)), caption="28×28 normalized image", clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Class probabilities
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Class Probabilities")
        chart = prob_bar_chart(pred_probs)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download predictions
        result = {
            "predicted_digit": pred_digit,
            "confidence": pred_conf,
            "probabilities": {str(i): float(p) for i, p in enumerate(pred_probs)},
        }
        st.download_button(
            label="📥 Download Prediction (JSON)",
            data=json.dumps(result, indent=2),
            file_name="mnist_prediction.json",
            mime="application/json",
        )
    else:
        st.info("Draw or upload a digit, then click **Predict** to see the result.")


st.markdown("---")
st.caption("Built with Streamlit + TensorFlow • Advanced MNIST Digit Recognition App")
