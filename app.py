"""
Streamlit interactive web application for MNIST handwritten digit recognition.
Provides dual-mode input (drawing canvas & file upload), real-time CNN prediction,
class probability distribution, and 28x28 preprocessed tensor preview.
"""

from pathlib import Path
from PIL import Image
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch

from src.inference import DigitPredictor
from src.utils import get_device, load_config

# Page Configuration
st.set_page_config(
    page_title="MNIST Digit Classifier | Deep CNN",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Navy & Slate Theme)
st.markdown(
    """
    <style>
        /* Overall Page Font & Colors */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Metric Card Container */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 16px;
        }
        
        /* Dominant Predicted Digit */
        .predicted-digit {
            font-size: 80px;
            font-weight: 800;
            color: #3B82F6;
            line-height: 1.0;
            margin: 8px 0 16px 0;
            text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
        }
        
        /* Confidence Badges */
        .confidence-badge-high {
            display: inline-block;
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid #10B981;
            border-radius: 20px;
            padding: 6px 18px;
            font-weight: 600;
            font-size: 15px;
        }
        
        .confidence-badge-med {
            display: inline-block;
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid #F59E0B;
            border-radius: 20px;
            padding: 6px 18px;
            font-weight: 600;
            font-size: 15px;
        }
        
        .confidence-badge-low {
            display: inline-block;
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid #EF4444;
            border-radius: 20px;
            padding: 6px 18px;
            font-weight: 600;
            font-size: 15px;
        }
        
        /* Previews and Panels */
        .preview-panel {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }
        
        .subtle-label {
            color: #94A3B8;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_digit_predictor() -> DigitPredictor:
    """Initialize and cache the model predictor in memory."""
    checkpoint = "models/best_model.pth"
    return DigitPredictor(checkpoint_path=checkpoint)


def render_probability_chart(probabilities: dict, predicted_digit: int):
    """Render horizontal probability distribution using Altair."""
    data = []
    for digit_str, prob in probabilities.items():
        digit_int = int(digit_str)
        data.append({
            "Digit": str(digit_int),
            "Probability": prob,
            "Percentage": f"{prob * 100:.1f}%",
            "IsTop": (digit_int == predicted_digit),
        })
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Predicted Probability", grid=True, gridColor="#334155"),
            ),
            y=alt.Y(
                "Digit:N",
                sort=list(map(str, range(10))),
                axis=alt.Axis(title="Digit Class (0–9)", labelFontSize=12, titleFontSize=12),
            ),
            color=alt.Color(
                "IsTop:N",
                scale=alt.Scale(domain=[True, False], range=["#3B82F6", "#475569"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Digit:N", title="Digit"),
                alt.Tooltip("Percentage:N", title="Probability"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    # Load Predictor
    try:
        predictor = get_digit_predictor()
    except Exception as e:
        st.error(f"❌ Failed to load model checkpoint: {str(e)}")
        st.stop()

    device_name = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if predictor.device.type == "cuda"
        else "CPU"
    )

    # -------------------------------------------------------------
    # SIDEBAR: Model Metadata & Specifications
    # -------------------------------------------------------------
    with st.sidebar:
        st.title("🔢 MNIST Classifier")
        st.markdown(
            "An internship-ready deep learning system for handwritten digit recognition implemented in **PyTorch**."
        )

        st.markdown("---")
        st.markdown("### 📊 Model Specifications")
        st.markdown(
            """
            * **Architecture**: Dual-Block CNN
            * **Regularization**: BatchNorm2d + Dropout
            * **Parameters**: 468,458 trainable
            * **Training Set**: 50,000 samples
            * **Validation Set**: 10,000 samples
            * **Canonical Test Set**: 10,000 samples
            * **Measured Test Accuracy**: **99.64%**
            * **Measured Test F1**: **0.9964**
            """
        )

        st.markdown("---")
        st.markdown("### ⚡ Runtime Hardware")
        st.info(f"**Inference Device**:\n`{device_name}`")

        st.markdown("---")
        st.markdown("### ⚠️ Input Scope")
        st.caption(
            "This classifier is designed for **one isolated handwritten digit at a time (0–9)**. "
            "Multi-digit segmentation is not supported. "
            "Note: 99.64% accuracy was measured on canonical MNIST; external drawing performance may vary due to domain shift."
        )

    # -------------------------------------------------------------
    # MAIN HEADER
    # -------------------------------------------------------------
    st.title("MNIST Digit Classifier")
    st.markdown(
        "Draw or upload a single handwritten digit (**0–9**) and let the Convolutional Neural Network predict it in real-time."
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # INPUT SECTION: Draw or Upload
    # -------------------------------------------------------------
    tab_draw, tab_upload = st.tabs(["✏️ Draw Digit", "📤 Upload Image"])

    active_image = None
    input_source = None

    with tab_draw:
        col_canvas, col_canvas_ctrl = st.columns([1, 1], gap="medium")

        with col_canvas:
            st.markdown("<div class='subtle-label'>Interactive Drawing Canvas</div>", unsafe_allow_html=True)
            stroke_width = st.slider("Brush Stroke Width", min_value=10, max_value=28, value=18, step=2)

            # Interactive canvas: Black background with white strokes
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )

        with col_canvas_ctrl:
            st.markdown("<div class='subtle-label'>Canvas Instructions</div>", unsafe_allow_html=True)
            st.markdown(
                """
                1. Draw a single digit (**0–9**) in the box.
                2. Use bold, clear strokes near the center.
                3. Click **Predict Drawn Digit** below.
                4. Use the canvas trash icon to clear.
                """
            )
            btn_predict_draw = st.button("✨ Predict Drawn Digit", type="primary", use_container_width=True)

            if btn_predict_draw:
                if canvas_result is not None and canvas_result.image_data is not None:
                    active_image = canvas_result.image_data
                    input_source = "Canvas Drawing"

    with tab_upload:
        col_up_input, col_up_preview = st.columns([1, 1], gap="medium")

        with col_up_input:
            st.markdown("<div class='subtle-label'>Upload Handwritten Image</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg", "webp"],
                help="Supports black-on-white paper drawings or white-on-black digital scans.",
            )
            btn_predict_upload = st.button("✨ Predict Uploaded Digit", type="primary", use_container_width=True)

        with col_up_preview:
            if uploaded_file is not None:
                st.markdown("<div class='subtle-label'>Uploaded Image Preview</div>", unsafe_allow_html=True)
                try:
                    pil_img = Image.open(uploaded_file)
                    st.image(pil_img, width=180, caption=f"Original: {uploaded_file.name}")
                    if btn_predict_upload:
                        active_image = pil_img
                        input_source = f"File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Could not open uploaded image: {e}")

    # -------------------------------------------------------------
    # PREDICTION EXECUTION & DISPLAY
    # -------------------------------------------------------------
    if active_image is not None:
        with st.spinner("Processing image and executing CNN forward pass..."):
            try:
                result = predictor.predict(active_image)
                st.session_state["last_result"] = result
                st.session_state["last_source"] = input_source
            except ValueError as ve:
                st.warning(f"⚠️ {str(ve)}")
                st.session_state["last_result"] = None
            except Exception as e:
                st.error(f"❌ An error occurred during inference: {str(e)}")
                st.session_state["last_result"] = None

    # Render results if available in session state
    if st.session_state.get("last_result") is not None:
        res = st.session_state["last_result"]
        source = st.session_state.get("last_source", "Input")

        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        st.caption(f"Evaluated from: **{source}**")

        pred_digit = res["digit"]
        confidence = res["confidence"]
        conf_pct = confidence * 100

        # Confidence categorization
        if conf_pct >= 90.0:
            badge_html = f"<div class='confidence-badge-high'>High Confidence: {conf_pct:.2f}%</div>"
            advice = None
        elif conf_pct >= 70.0:
            badge_html = f"<div class='confidence-badge-med'>Moderate Confidence: {conf_pct:.2f}%</div>"
            advice = "The model is moderately confident. Try drawing slightly thicker or clearer strokes."
        else:
            badge_html = f"<div class='confidence-badge-low'>Low Confidence: {conf_pct:.2f}%</div>"
            advice = "⚠️ The model is less confident in this prediction. Try drawing the digit more clearly or centering your stroke."

        col_result_card, col_prob_dist = st.columns([1, 1], gap="large")

        with col_result_card:
            # Dominant result card
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='subtle-label'>Model Classification</div>
                    <div class='predicted-digit'># {pred_digit}</div>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if advice:
                st.info(advice)

            # Processed 28x28 Image Preview
            st.markdown("<div class='subtle-label'>Processed Input (28×28 Tensor)</div>", unsafe_allow_html=True)
            st.image(
                res["processed_image"],
                width=160,
                clamp=True,
                caption="Normalized, Centered 28×28 Representation used by CNN",
            )
            st.caption(
                "This preview confirms that the background was inverted, the bounding box was cropped to 20×20, "
                "and the digit's center of mass was shifted to the canvas center."
            )

        with col_prob_dist:
            st.markdown("<div class='subtle-label'>Full Class Probability Distribution</div>", unsafe_allow_html=True)
            render_probability_chart(res["probabilities"], pred_digit)

    # -------------------------------------------------------------
    # EXPLAINABILITY / PIPELINE EXPLANATION
    # -------------------------------------------------------------
    st.markdown("---")
    with st.expander("ℹ️ How the Preprocessing Pipeline Works", expanded=False):
        st.markdown(
            """
            External handwritten digits often fail on standard MNIST models due to differences in stroke width,
            contrast inversion, and off-center placement. This application runs an automated preprocessing pipeline
            prior to feeding data into the CNN:

            ```text
            Draw / Upload Image
                    ↓
            1. Grayscale & Intensity Scaling     (Converts input to single-channel float [0.0, 1.0])
                    ↓
            2. Polarity Detection & Inversion   (Detects white-on-black vs black-on-white)
                    ↓
            3. Blank & Uniform Rejection        (Rejects blank canvases; prevents random guessing)
                    ↓
            4. Bounding-Box Detection & Crop    (Isolates active stroke contours)
                    ↓
            5. Aspect-Ratio-Preserving Resize   (Fits maximum dimension to 20 pixels)
                    ↓
            6. Center-of-Mass Alignment         (Shifts centroid to (13.5, 13.5) via image moments)
                    ↓
            7. MNIST Z-Score Normalization      (Standardizes using μ=0.1307, σ=0.3081)
                    ↓
            8. CNN Forward Pass & Softmax       (Calculates logits, then extracts probabilities)
            ```
            """
        )


if __name__ == "__main__":
    main()
