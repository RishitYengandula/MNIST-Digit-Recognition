"""
Unit tests for real-world inference preprocessing and prediction pipeline.
Tests PIL, NumPy, RGBA/RGB/Grayscale formats, blank rejection, polarity detection,
and numerical safety.
"""

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from src.inference import DigitPredictor
from src.utils import get_device, set_seed


@pytest.fixture(scope="module")
def predictor():
    """Instantiate DigitPredictor once with best checkpoint."""
    set_seed(42)
    return DigitPredictor("models/best_model.pth")


@pytest.fixture
def synthetic_digit_images():
    """Generate various formats of a synthetic digit '1'."""
    # 1. PIL Grayscale '1' (White background, black stroke)
    img_pil_l = Image.new("L", (100, 100), color=255)
    draw_l = ImageDraw.Draw(img_pil_l)
    draw_l.line([(50, 15), (50, 85)], fill=0, width=10)

    # 2. PIL RGB '1'
    img_pil_rgb = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw_rgb = ImageDraw.Draw(img_pil_rgb)
    draw_rgb.line([(50, 15), (50, 85)], fill=(0, 0, 0), width=10)

    # 3. PIL RGBA '1' (Black stroke on transparent background)
    img_pil_rgba = Image.new("RGBA", (100, 100), color=(0, 0, 0, 0))
    draw_rgba = ImageDraw.Draw(img_pil_rgba)
    draw_rgba.line([(50, 15), (50, 85)], fill=(0, 0, 0, 255), width=10)

    # 4. NumPy 2D grayscale (Black background, white stroke)
    np_2d = np.zeros((100, 100), dtype=np.float32)
    np_2d[15:85, 45:55] = 1.0

    # 5. NumPy 3D RGB (White background, black stroke)
    np_rgb = np.ones((100, 100, 3), dtype=np.uint8) * 255
    np_rgb[15:85, 45:55, :] = 0

    # 6. NumPy 3D RGBA
    np_rgba = np.zeros((100, 100, 4), dtype=np.float32)
    np_rgba[15:85, 45:55, :3] = 0.0
    np_rgba[15:85, 45:55, 3] = 1.0

    return {
        "pil_l": img_pil_l,
        "pil_rgb": img_pil_rgb,
        "pil_rgba": img_pil_rgba,
        "np_2d": np_2d,
        "np_rgb": np_rgb,
        "np_rgba": np_rgba,
    }


def test_blank_image_rejection(predictor):
    """Confirm completely blank images raise a clean ValueError."""
    # Uniform white PIL image
    blank_white = Image.new("RGB", (80, 80), color=(255, 255, 255))
    with pytest.raises(ValueError, match="No meaningful digit detected"):
        predictor.predict(blank_white)

    # Uniform black NumPy array
    blank_black = np.zeros((80, 80), dtype=np.float32)
    with pytest.raises(ValueError, match="No meaningful digit detected"):
        predictor.predict(blank_black)

    # Very faint noise (below contrast threshold)
    faint_noise = np.ones((80, 80), dtype=np.float32) * 0.05
    with pytest.raises(ValueError, match="No meaningful digit detected"):
        predictor.predict(faint_noise)


def test_supported_input_formats(predictor, synthetic_digit_images):
    """Verify inference pipeline processes PIL (L, RGB, RGBA) and NumPy (2D, 3D) without errors."""
    for format_name, img_data in synthetic_digit_images.items():
        result = predictor.predict(img_data)
        assert isinstance(result, dict), f"Failed for format {format_name}"
        assert "digit" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "processed_image" in result
        assert 0 <= result["digit"] <= 9


def test_preprocessed_output_shapes_and_ranges(predictor, synthetic_digit_images):
    """Verify preprocessed tensor has shape (1, 1, 28, 28) and unnormalized display image is (28, 28)."""
    img = synthetic_digit_images["pil_rgb"]
    tensor_norm, canvas_display = predictor.preprocess(img)

    assert tensor_norm.shape == (1, 1, 28, 28)
    assert isinstance(tensor_norm, torch.Tensor)
    assert not torch.isnan(tensor_norm).any()
    assert not torch.isinf(tensor_norm).any()

    assert canvas_display.shape == (28, 28)
    assert isinstance(canvas_display, np.ndarray)
    assert 0.0 <= canvas_display.min() <= 0.1
    assert 0.8 <= canvas_display.max() <= 1.0


def test_probabilities_properties(predictor, synthetic_digit_images):
    """Ensure class probabilities sum to ~1.0, confidence matches argmax, and values in [0, 1]."""
    result = predictor.predict(synthetic_digit_images["pil_l"])

    probs = result["probabilities"]
    assert len(probs) == 10

    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 0.02, f"Probabilities sum to {prob_sum}, expected ~1.0"

    predicted_digit_str = str(result["digit"])
    assert result["confidence"] == probs[predicted_digit_str]
    assert probs[predicted_digit_str] == max(probs.values())


def test_polarity_inversion_handling(predictor):
    """Verify model correctly classifies both black-on-white and white-on-black formats."""
    # Digit '0' drawn on white background with black stroke
    img_bw = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw_bw = ImageDraw.Draw(img_bw)
    draw_bw.ellipse([25, 20, 75, 80], outline=(0, 0, 0), width=10)
    res_bw = predictor.predict(img_bw)

    # Digit '0' drawn on black background with white stroke
    img_wb = Image.new("L", (100, 100), color=0)
    draw_wb = ImageDraw.Draw(img_wb)
    draw_wb.ellipse([25, 20, 75, 80], outline=255, width=10)
    res_wb = predictor.predict(img_wb)

    assert res_bw["digit"] == 0, f"Expected 0 for black-on-white, got {res_bw['digit']}"
    assert res_wb["digit"] == 0, f"Expected 0 for white-on-black, got {res_wb['digit']}"


def test_cpu_and_cuda_inference(synthetic_digit_images):
    """Confirm DigitPredictor can run explicitly on CPU and CUDA."""
    img = synthetic_digit_images["pil_rgb"]

    # Explicit CPU predictor
    cpu_predictor = DigitPredictor(checkpoint_path="models/best_model.pth", device=torch.device("cpu"))
    res_cpu = cpu_predictor.predict(img)
    assert 0 <= res_cpu["digit"] <= 9

    # Explicit CUDA predictor if available
    if torch.cuda.is_available():
        cuda_predictor = DigitPredictor(checkpoint_path="models/best_model.pth", device=torch.device("cuda"))
        res_cuda = cuda_predictor.predict(img)
        assert res_cuda["digit"] == res_cpu["digit"]
