"""
Inference and image preprocessing pipeline for MNIST digit recognition.
Handles external image inputs (PIL / NumPy), polarity inversion, bounding-box cropping,
aspect-ratio-preserving resizing, and center-of-mass alignment into a 28x28 canvas.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.model import MNISTCNN
from src.train import load_checkpoint
from src.utils import get_device, load_config


class DigitPredictor:
    """
    Production-quality inference engine for handwritten digit recognition.

    Adapts arbitrary external images (e.g., from drawing canvas or camera uploads)
    to match the MNIST input distribution:
      1. Converts to grayscale.
      2. Normalizes pixel values to [0.0, 1.0].
      3. Automatically detects background/foreground polarity and inverts if necessary.
      4. Detects foreground digit bounding box; raises ValueError if blank or no digit found.
      5. Resizes digit preserving aspect ratio to fit within a 20x20 box.
      6. Embeds within a 28x28 canvas and centers using the digit's center of mass.
      7. Normalizes using canonical MNIST mean (0.1307) and std (0.3081).
      8. Computes raw CNN logits, then applies Softmax to yield class probabilities.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/best_model.pth",
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        if config is None:
            try:
                self.config = load_config()
            except Exception:
                self.config = {}
        else:
            self.config = config

        norm_cfg = self.config.get("dataset", {}).get("normalization", {})
        self.mean = float(norm_cfg.get("mean", [0.1307])[0])
        self.std = float(norm_cfg.get("std", [0.3081])[0])

        self.model, self.checkpoint_meta = load_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=self.device,
        )
        self.model.eval()

    def _convert_to_grayscale_array(
        self,
        image_input: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """
        Convert PIL Image or NumPy array input into a single-channel float32 NumPy array in [0.0, 1.0].
        """
        if isinstance(image_input, Image.Image):
            # Handle RGBA with transparency
            if image_input.mode == "RGBA":
                alpha = np.array(image_input)[:, :, 3]
                rgb = np.array(image_input)[:, :, :3]
                # If transparent areas exist, composite onto background depending on stroke tone
                visible_mask = alpha > 30
                if np.any(visible_mask):
                    mean_stroke_intensity = np.mean(rgb[visible_mask])
                    bg_color = (0, 0, 0) if mean_stroke_intensity > 128 else (255, 255, 255)
                else:
                    bg_color = (255, 255, 255)

                bg_image = Image.new("RGB", image_input.size, bg_color)
                bg_image.paste(image_input, mask=image_input.split()[3])
                gray_img = bg_image.convert("L")
            else:
                gray_img = image_input.convert("L")

            arr = np.array(gray_img, dtype=np.float32) / 255.0

        elif isinstance(image_input, np.ndarray):
            arr = image_input.astype(np.float32)

            # If pixel values are in integer 0-255 range, scale to 0.0-1.0
            if arr.max() > 1.0:
                arr = arr / 255.0

            # Convert 3D arrays (H, W, C) to 2D grayscale
            if arr.ndim == 3:
                channels = arr.shape[2]
                if channels == 4:  # RGBA
                    alpha = arr[:, :, 3]
                    rgb = arr[:, :, :3]
                    visible_mask = alpha > 0.1
                    if np.any(visible_mask):
                        mean_stroke = np.mean(rgb[visible_mask])
                        bg_val = 0.0 if mean_stroke > 0.5 else 1.0
                    else:
                        bg_val = 1.0
                    composite = rgb * alpha[:, :, np.newaxis] + bg_val * (1.0 - alpha[:, :, np.newaxis])
                    arr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                elif channels == 3:  # RGB
                    arr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                elif channels == 1:
                    arr = arr.squeeze(2)
                else:
                    raise ValueError(f"Unsupported number of image channels: {channels}")
            elif arr.ndim != 2:
                raise ValueError(f"Expected 2D or 3D NumPy array, got shape {arr.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected PIL.Image or np.ndarray.")

        return np.clip(arr, 0.0, 1.0)

    def preprocess(
        self,
        image_input: Union[Image.Image, np.ndarray],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Execute full preprocessing pipeline:
          1. Convert to grayscale float array.
          2. Detect polarity and invert if background is bright.
          3. Validate non-blank image.
          4. Crop bounding box enclosing the digit.
          5. Resize preserving aspect ratio into 20x20 box.
          6. Center on 28x28 canvas using center of mass.
          7. Normalize to MNIST tensor.

        Returns:
            Tuple[torch.Tensor, np.ndarray]:
              - Normalized PyTorch tensor of shape (1, 1, 28, 28)
              - Preprocessed 28x28 NumPy array in [0.0, 1.0] (for visual display)

        Raises:
            ValueError: If no meaningful foreground digit is detected.
        """
        gray = self._convert_to_grayscale_array(image_input)

        # 1. Determine background/foreground polarity
        # Sample border pixels (top, bottom, left, right edges)
        border_pixels = np.concatenate([
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ])
        border_mean = float(np.mean(border_pixels))

        # If border is bright (> 0.5), user drew dark strokes on light paper/canvas -> invert
        if border_mean > 0.5:
            img = 1.0 - gray
        else:
            img = gray.copy()

        # 2. Blank image & faint stroke detection
        # Contrast check and max intensity check
        max_val = float(np.max(img))
        min_val = float(np.min(img))
        contrast = max_val - min_val

        # Require a minimum foreground intensity and contrast
        if max_val < 0.15 or contrast < 0.15:
            raise ValueError("No meaningful digit detected in the input image.")

        # Threshold to identify foreground digit pixels
        # Use dynamic threshold slightly above the baseline minimum
        thresh_val = max(0.12, min_val + 0.15 * contrast)
        binary_mask = img > thresh_val

        # Count active foreground pixels
        active_pixel_count = int(np.sum(binary_mask))
        if active_pixel_count < 12:
            raise ValueError("No meaningful digit detected in the input image.")

        # 3. Detect bounding box enclosing the foreground
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop tightly around foreground
        cropped = img[rmin : rmax + 1, cmin : cmax + 1]
        crop_h, crop_w = cropped.shape

        if crop_h == 0 or crop_w == 0:
            raise ValueError("No meaningful digit detected in the input image.")

        # 4. Aspect-ratio-preserving resize into target 20x20 box
        target_box_dim = 20.0
        if crop_h > crop_w:
            scale = target_box_dim / float(crop_h)
            new_h = int(target_box_dim)
            new_w = max(1, int(round(crop_w * scale)))
        else:
            scale = target_box_dim / float(crop_w)
            new_w = int(target_box_dim)
            new_h = max(1, int(round(crop_h * scale)))

        # High-quality interpolation
        interp = cv2.INTER_AREA if (new_w < crop_w or new_h < crop_h) else cv2.INTER_CUBIC
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)
        resized = np.clip(resized, 0.0, 1.0)

        # 5. Place in 28x28 canvas (initial bounding-box centering)
        canvas = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        # 6. Fine alignment by Center of Mass (Image Moments)
        # MNIST digits were centered by their center of mass rather than geometric bounding box.
        mass = float(np.sum(canvas))
        if mass > 0:
            grid_y, grid_x = np.indices((28, 28), dtype=np.float32)
            cy = float(np.sum(grid_y * canvas) / mass)
            cx = float(np.sum(grid_x * canvas) / mass)

            # Target center is (13.5, 13.5)
            dx = 13.5 - cx
            dy = 13.5 - cy

            # Limit shift magnitude to [-4, 4] pixels to prevent clipping digits against borders
            dx = float(np.clip(dx, -4.0, 4.0))
            dy = float(np.clip(dy, -4.0, 4.0))

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(
                canvas,
                M,
                (28, 28),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        # Final unnormalized 28x28 image for display
        canvas_display = np.clip(canvas, 0.0, 1.0)

        # 7. Convert to normalized PyTorch tensor (1, 1, 28, 28)
        tensor = torch.from_numpy(canvas_display).unsqueeze(0).unsqueeze(0).float()
        tensor_normalized = (tensor - self.mean) / self.std

        return tensor_normalized, canvas_display

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Execute end-to-end digit prediction on an external image.

        Args:
            image_input: PIL Image or NumPy array.

        Returns:
            Dict[str, Any]:
              - "digit": Predicted integer digit (0-9)
              - "confidence": Float probability of predicted class in [0.0, 1.0]
              - "probabilities": Dict mapping digit strings "0"-"9" to probabilities
              - "processed_image": 28x28 NumPy array in [0.0, 1.0]
              - "input_tensor": (1, 1, 28, 28) normalized PyTorch tensor

        Raises:
            ValueError: If image is blank or has no meaningful digit.
        """
        tensor_norm, canvas_display = self.preprocess(image_input)

        with torch.no_grad():
            tensor_dev = tensor_norm.to(self.device)
            logits = self.model(tensor_dev)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # Shape: (10,)

        conf_tensor, pred_tensor = torch.max(probs, dim=0)
        pred_digit = int(pred_tensor.item())
        confidence = float(conf_tensor.item())

        prob_dict = {str(i): round(float(probs[i].item()), 4) for i in range(10)}

        return {
            "digit": pred_digit,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "processed_image": canvas_display,
            "input_tensor": tensor_norm,
        }


if __name__ == "__main__":
    from PIL import ImageDraw

    print("=" * 60)
    print("MNIST Digit Predictor — Inference Test")
    print("=" * 60)

    predictor = DigitPredictor()
    print(f"Device: {predictor.device.type.upper()}")

    # 1. Synthetic digit '7' on white canvas (black strokes)
    img_7 = Image.new("RGB", (120, 120), color=(255, 255, 255))
    draw_7 = ImageDraw.Draw(img_7)
    draw_7.line([(30, 25), (90, 25)], fill=(0, 0, 0), width=10)
    draw_7.line([(90, 25), (50, 95)], fill=(0, 0, 0), width=10)

    res_7 = predictor.predict(img_7)
    print("\nTest 1 — Synthetic Digit '7' (Black on White):")
    print(f"  - Predicted Digit: {res_7['digit']}")
    print(f"  - Confidence:      {res_7['confidence'] * 100:.2f}%")
    print(f"  - Top Probabilities: {sorted(res_7['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]}")

    # 2. Synthetic digit '0' on black canvas (white strokes)
    img_0 = Image.new("L", (100, 100), color=0)
    draw_0 = ImageDraw.Draw(img_0)
    draw_0.ellipse([25, 20, 75, 80], outline=255, width=8)

    res_0 = predictor.predict(img_0)
    print("\nTest 2 — Synthetic Digit '0' (White on Black):")
    print(f"  - Predicted Digit: {res_0['digit']}")
    print(f"  - Confidence:      {res_0['confidence'] * 100:.2f}%")
    print(f"  - Top Probabilities: {sorted(res_0['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]}")

    # 3. Blank image test (Should raise ValueError)
    blank_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    try:
        predictor.predict(blank_img)
        print("\nTest 3 — Blank Image: FAILED (Expected ValueError)")
    except ValueError as e:
        print(f"\nTest 3 — Blank Image: PASSED (Raised ValueError: '{e}')")

    print("=" * 60)
