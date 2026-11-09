"""
Convolutional Neural Network (CNN) architecture definition for MNIST digit recognition.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class MNISTCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST handwritten digit recognition.

    Architecture:
      Conv Block 1:
        - Conv2d(1 -> 32, kernel_size=3, padding=1)
        - BatchNorm2d(32)
        - ReLU
        - Conv2d(32 -> 32, kernel_size=3, padding=1)
        - BatchNorm2d(32)
        - ReLU
        - MaxPool2d(kernel_size=2, stride=2) -> (B, 32, 14, 14)
        - Dropout2d(p=0.25)

      Conv Block 2:
        - Conv2d(32 -> 64, kernel_size=3, padding=1)
        - BatchNorm2d(64)
        - ReLU
        - Conv2d(64 -> 64, kernel_size=3, padding=1)
        - BatchNorm2d(64)
        - ReLU
        - MaxPool2d(kernel_size=2, stride=2) -> (B, 64, 7, 7)
        - Dropout2d(p=0.25)

      Classification Head:
        - Flatten -> (B, 3136)
        - Linear(3136 -> 128)
        - BatchNorm1d(128)
        - ReLU
        - Dropout(p=0.5)
        - Linear(128 -> 10) -> Raw Logits (B, 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_conv: float = 0.25,
        dropout_fc: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
        )

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
        )

        # Classification Head: 64 channels * 7 height * 7 width = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding unnormalized class logits.

        Args:
            x: Input tensor of shape (Batch, 1, 28, 28)

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 10). Softmax is NOT applied here.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Calculate the exact number of trainable parameters in an nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_parameter_summary(model: nn.Module) -> Dict[str, int]:
    """Return a breakdown of parameter counts by named submodule."""
    summary = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary[name] = count
    return summary


if __name__ == "__main__":
    from src.utils import get_device

    print("=" * 60)
    print("MNIST CNN Architecture Verification")
    print("=" * 60)

    model = MNISTCNN()
    total_params = count_parameters(model)
    layer_summary = get_layer_parameter_summary(model)

    print(model)
    print("\nParameter Breakdown:")
    for name, count in layer_summary.items():
        print(f"  - {name:<16}: {count:>8,} trainable parameters")
    print(f"  - {'TOTAL':<16}: {total_params:>8,} trainable parameters")

    # CPU verification
    dummy_input = torch.randn(4, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        out_cpu = model(dummy_input)
    print(f"\nCPU Forward Pass:")
    print(f"  - Input Shape:  {list(dummy_input.shape)}")
    print(f"  - Output Shape: {list(out_cpu.shape)}")
    print(f"  - Sample Logits:\n{out_cpu[0].numpy()}")
    print(f"  - Logit Sum:    {out_cpu[0].sum().item():.4f} (Not forced to 1.0, raw logits verified)")

    # CUDA verification if available
    device = get_device()
    if device.type == "cuda":
        model_cuda = MNISTCNN().to(device)
        model_cuda.eval()
        dummy_cuda = dummy_input.to(device)
        with torch.no_grad():
            out_cuda = model_cuda(dummy_cuda)
        print(f"\nCUDA Forward Pass ({torch.cuda.get_device_name(0)}):")
        print(f"  - Input Device:  {dummy_cuda.device}")
        print(f"  - Output Device: {out_cuda.device}")
        print(f"  - Output Shape:  {list(out_cuda.shape)}")
    print("=" * 60)
