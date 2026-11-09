"""
Unit tests for the MNIST Convolutional Neural Network (CNN) architecture.
"""

import pytest
import torch
import torch.nn as nn

from src.model import MNISTCNN, count_parameters, get_layer_parameter_summary
from src.utils import get_device, set_seed


@pytest.fixture(autouse=True)
def init_seed():
    """Ensure deterministic initialization for all model tests."""
    set_seed(42)


def test_forward_pass_shape():
    """Verify input shape (4, 1, 28, 28) produces output shape (4, 10)."""
    model = MNISTCNN()
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 10), f"Expected output shape (4, 10), got {out.shape}"
    assert out.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [1, 8, 16, 64])
def test_batch_size_flexibility(batch_size):
    """Confirm model accepts variable batch sizes including edge case batch_size=1."""
    model = MNISTCNN()
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 10), f"Failed for batch_size={batch_size}, got {out.shape}"


def test_output_is_finite():
    """Assert output contains no NaN or Inf values across multiple forward passes."""
    model = MNISTCNN()
    model.eval()
    x = torch.randn(16, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    assert torch.isfinite(out).all(), "Non-finite values (NaN or Inf) detected in model output"
    assert not torch.isnan(out).any(), "NaN values found in model output"
    assert not torch.isinf(out).any(), "Inf values found in model output"


def test_parameter_count_exact():
    """Calculate and verify exact trainable parameter counts layer-by-layer."""
    model = MNISTCNN()
    total_params = count_parameters(model)
    layer_summary = get_layer_parameter_summary(model)

    # Exact expected calculations:
    # Conv Block 1:
    #   Conv2d(1, 32, 3, p=1): 32*1*9 + 32 = 320
    #   BatchNorm2d(32): 32*2 = 64
    #   Conv2d(32, 32, 3, p=1): 32*32*9 + 32 = 9248
    #   BatchNorm2d(32): 32*2 = 64
    #   Subtotal = 9,696
    # Conv Block 2:
    #   Conv2d(32, 64, 3, p=1): 64*32*9 + 64 = 18496
    #   BatchNorm2d(64): 64*2 = 128
    #   Conv2d(64, 64, 3, p=1): 64*64*9 + 64 = 36928
    #   BatchNorm2d(64): 64*2 = 128
    #   Subtotal = 55,680
    # Classifier Head:
    #   Linear(3136, 128): 3136*128 + 128 = 401536
    #   BatchNorm1d(128): 128*2 = 256
    #   Linear(128, 10): 128*10 + 10 = 1290
    #   Subtotal = 403,082
    # Grand Total = 9,696 + 55,680 + 403,082 = 468,458
    assert layer_summary["conv_block1"] == 9696
    assert layer_summary["conv_block2"] == 55680
    assert layer_summary["classifier"] == 403082
    assert total_params == 468458, f"Expected 468,458 trainable parameters, got {total_params}"


def test_model_major_components_exist():
    """Verify that all required architectural layers are present in the model."""
    model = MNISTCNN()
    modules = list(model.modules())

    has_conv2d = any(isinstance(m, nn.Conv2d) for m in modules)
    has_batchnorm2d = any(isinstance(m, nn.BatchNorm2d) for m in modules)
    has_batchnorm1d = any(isinstance(m, nn.BatchNorm1d) for m in modules)
    has_maxpool2d = any(isinstance(m, nn.MaxPool2d) for m in modules)
    has_dropout2d = any(isinstance(m, nn.Dropout2d) for m in modules)
    has_dropout = any(isinstance(m, nn.Dropout) for m in modules)
    has_linear = any(isinstance(m, nn.Linear) for m in modules)
    has_relu = any(isinstance(m, nn.ReLU) for m in modules)

    assert has_conv2d, "Model missing nn.Conv2d"
    assert has_batchnorm2d, "Model missing nn.BatchNorm2d"
    assert has_batchnorm1d, "Model missing nn.BatchNorm1d"
    assert has_maxpool2d, "Model missing nn.MaxPool2d"
    assert has_dropout2d, "Model missing nn.Dropout2d"
    assert has_dropout, "Model missing nn.Dropout"
    assert has_linear, "Model missing nn.Linear"
    assert has_relu, "Model missing nn.ReLU"


def test_output_is_raw_logits_not_probabilities():
    """Verify final outputs are raw logits (not normalized to probabilities via Softmax)."""
    model = MNISTCNN()
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    # 1. Softmax outputs are strictly >= 0; raw logits can be negative
    has_negative_values = (out < 0).any().item()
    assert has_negative_values, "Expected raw logits to contain negative values"

    # 2. Softmax outputs strictly sum to 1.0 along class dimension; raw logits do not
    sums = out.sum(dim=1)
    is_sum_not_one = not torch.allclose(sums, torch.ones_like(sums), atol=1e-2)
    assert is_sum_not_one, f"Outputs unexpectedly sum to 1.0: {sums}"


def test_device_movement_and_forward_pass():
    """Verify model moves to detected device (CPU or CUDA) and executes forward pass."""
    # Test CPU explicitly
    cpu_device = torch.device("cpu")
    model_cpu = MNISTCNN().to(cpu_device)
    model_cpu.eval()
    x_cpu = torch.randn(2, 1, 28, 28, device=cpu_device)
    with torch.no_grad():
        out_cpu = model_cpu(x_cpu)
    assert out_cpu.device.type == "cpu"
    assert out_cpu.shape == (2, 10)

    # Test CUDA if available
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        model_cuda = MNISTCNN().to(cuda_device)
        model_cuda.eval()
        x_cuda = torch.randn(2, 1, 28, 28, device=cuda_device)
        with torch.no_grad():
            out_cuda = model_cuda(x_cuda)
        assert out_cuda.device.type == "cuda"
        assert out_cuda.shape == (2, 10)


def test_eval_mode_behavior():
    """Confirm model.eval() sets deterministic execution (disables dropout randomness)."""
    model = MNISTCNN()
    model.eval()
    x = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    # In eval mode, two forward passes on identical input must be identical
    assert torch.allclose(out1, out2), "Model outputs differed in eval() mode"
