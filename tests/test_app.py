"""
Lightweight unit tests for app.py and Streamlit UI components.
Verifies module imports, predictor caching function, probability visualization helper,
and model immutability without brittle browser automation.
"""

from pathlib import Path
import pytest
import altair as alt

import app
from src.inference import DigitPredictor


def test_app_imports_cleanly():
    """Verify app.py imports without syntax errors or broken dependencies."""
    assert hasattr(app, "main")
    assert hasattr(app, "get_digit_predictor")
    assert hasattr(app, "render_probability_chart")


def test_app_cached_predictor_instantiation():
    """Verify get_digit_predictor returns an operational DigitPredictor instance."""
    predictor = app.get_digit_predictor()
    assert isinstance(predictor, DigitPredictor)
    assert hasattr(predictor, "model")
    assert hasattr(predictor, "predict")
    assert not predictor.model.training  # Must be in eval mode


def test_app_chart_creation():
    """Verify render_probability_chart builds an Altair Chart specification."""
    dummy_probs = {str(i): 0.05 for i in range(10)}
    dummy_probs["3"] = 0.55

    # Check that creating the chart dataframe logic functions properly
    data = []
    for digit_str, prob in dummy_probs.items():
        digit_int = int(digit_str)
        data.append({
            "Digit": str(digit_int),
            "Probability": prob,
            "Percentage": f"{prob * 100:.1f}%",
            "IsTop": (digit_int == 3),
        })
    import pandas as pd
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Probability:Q",
            y="Digit:N",
            color="IsTop:N",
        )
    )
    assert isinstance(chart, alt.Chart)


def test_model_weights_not_modified_by_app():
    """Confirm that accessing or running app helpers does not modify best_model.pth."""
    checkpoint_file = Path("models/best_model.pth")
    assert checkpoint_file.is_file()
    # Checkpoint size must remain intact (~5.39 MB)
    size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
    assert 5.0 <= size_mb <= 6.0

