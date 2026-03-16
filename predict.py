"""PyTorch inference entrypoint for mini OCR.

Author: David
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

from ocr.config.dict_encoding import DictEncoding
from ocr.config.loader import load_config
from ocr.model.crnn import MiniCRNN
from ocr.utils.ctc import ctc_decode
from ocr.utils.image_utils import resize_gray_image

LOGGER = logging.getLogger("predict")
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run OCR inference with a PyTorch .pt model.")
    parser.add_argument(
        "--model-type",
        default="print_digital",
        help="Model type used to resolve default config path.",
    )
    parser.add_argument(
        "--cfg-file",
        default=None,
        help="Config path. Default: ./config/config_<model-type>.yaml",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="PyTorch checkpoint path (.pt). Default from cfg.TEST.CRNN_MODEL.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--image-path",
        default="./",
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--invert-threshold",
        type=float,
        default=128.0,
        help="Invert image when average grayscale is below this threshold. Set negative to disable.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable AMP inference on CUDA.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup forward passes before timed inference on CUDA.",
    )
    parser.add_argument("legacy_model_type", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_precision", nargs="?", type=int, help=argparse.SUPPRESS)
    parser.add_argument("legacy_device", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_image_path", nargs="?", help=argparse.SUPPRESS)
    return parser.parse_args()


def apply_legacy_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Apply legacy positional arguments for backward compatibility."""
    if args.legacy_model_type:
        args.model_type = args.legacy_model_type
    if args.legacy_precision is not None:
        LOGGER.warning("Legacy precision argument is ignored in PyTorch inference.")
    if args.legacy_device:
        legacy_device = str(args.legacy_device).lower()
        args.device = "cpu" if legacy_device == "cpu" else "cuda"
    if args.legacy_image_path:
        args.image_path = args.legacy_image_path
    return args


def setup_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_config_path(args: argparse.Namespace) -> Path:
    """Resolve config path from arguments."""
    if args.cfg_file:
        return Path(args.cfg_file)
    return Path(f"./config/config_{args.model_type}.yaml")


def resolve_model_path(args: argparse.Namespace, cfg) -> Path:
    """Resolve model path from arguments and config."""
    if args.model_path:
        return Path(args.model_path)

    test_path = str(getattr(cfg.TEST, "CRNN_MODEL", "")).strip() if hasattr(cfg, "TEST") else ""
    if test_path:
        return Path(test_path)

    checkpoint_dir = Path(str(cfg.SOLVER.CHECKPOINT_DIR))
    checkpoint_name = str(cfg.SOLVER.CHECKPOINT_NAME)
    return checkpoint_dir / checkpoint_name


def choose_device(device_arg: str) -> torch.device:
    """Choose runtime device."""
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(path: Path) -> list[Path]:
    """List supported images from a file or directory."""
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        return [path]

    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise FileNotFoundError(f"Image path not found: {path}")


def build_model(cfg, charset_size: int, device: torch.device) -> MiniCRNN:
    """Build MiniCRNN model."""
    model = MiniCRNN(
        charset_size=charset_size,
        backbone=cfg.MODEL.BACKBONE,
        encoder_type=cfg.MODEL.ENCODER_TYPE,
        encoder_input_size=cfg.MODEL.ENCODER_INPUT_SIZE,
        encoder_hidden_size=cfg.MODEL.ENCODER_HIDDEN_SIZE,
        encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        encoder_bidirectional=cfg.MODEL.ENCODER_BIDIRECTIONAL,
        max_seq_len=cfg.MODEL.MAX_SEQ_LEN,
    ).to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model weights from checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format. Expected a state_dict mapping.")

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def preprocess_image(
    image: np.ndarray,
    input_h: int,
    input_w: int,
    pixel_mean: float,
    pixel_std: float,
    invert_threshold: float,
) -> torch.Tensor:
    """Resize, normalize, and convert image to tensor (1, H, W)."""
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape: {image.shape}")

    if invert_threshold >= 0 and float(np.mean(image)) < invert_threshold:
        image = cv2.bitwise_not(image)

    image = resize_gray_image(image, input_h=input_h, input_w=input_w)
    image = image.astype(np.float32) / 255.0
    image = (image - pixel_mean) / pixel_std
    return torch.from_numpy(image[None, :, :])


def batched(items: list[Path], batch_size: int):
    """Yield slices from a list with fixed batch size."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_inference(
    model: torch.nn.Module,
    image_paths: list[Path],
    dict_encoding: DictEncoding,
    device: torch.device,
    input_h: int,
    input_w: int,
    pixel_mean: float,
    pixel_std: float,
    invert_threshold: float,
    batch_size: int,
    use_amp: bool,
    warmup_iters: int,
) -> None:
    """Run OCR inference and print results."""
    if batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if warmup_iters > 0:
            warmup_batch_size = max(1, min(batch_size, len(image_paths)))
            warmup_batch = torch.zeros(
                (warmup_batch_size, 1, input_h, input_w),
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode():
                for _ in range(warmup_iters):
                    if use_amp:
                        with autocast(device_type="cuda"):
                            _ = model(warmup_batch)
                    else:
                        _ = model(warmup_batch)
            torch.cuda.synchronize()
            LOGGER.info(
                "CUDA warmup finished: %d iterations (batch=%d)",
                warmup_iters,
                warmup_batch_size,
            )

    for batch_paths in batched(image_paths, batch_size):
        tensors: list[torch.Tensor] = []
        valid_paths: list[Path] = []

        for image_path in batch_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                LOGGER.warning("Failed to read image: %s", image_path)
                continue

            tensor = preprocess_image(
                image=image,
                input_h=input_h,
                input_w=input_w,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                invert_threshold=invert_threshold,
            )
            tensors.append(tensor)
            valid_paths.append(image_path)

        if not tensors:
            continue

        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)

        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            if use_amp and device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(batch)
            else:
                logits = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        predictions = logits.argmax(dim=2).cpu().numpy()
        decoded_texts = ctc_decode(predictions, alphabet_encoding=dict_encoding.dicts)

        per_sample_ms = elapsed_ms / max(len(decoded_texts), 1)
        for image_path, text in zip(valid_paths, decoded_texts):
            print(f"{image_path.name}\t{text}\t{per_sample_ms:.3f} ms")


def main() -> None:
    """Main entrypoint."""
    setup_logging()
    args = apply_legacy_overrides(parse_args())

    config_path = resolve_config_path(args)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    model_path = resolve_model_path(args, cfg)
    device = choose_device(args.device)

    dict_encoding = DictEncoding(cfg.DATASETS.CHAR_DICT)
    model = build_model(cfg, charset_size=len(dict_encoding.dicts) + 1, device=device)
    load_checkpoint(model, model_path, device)

    image_paths = list_images(Path(args.image_path))
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Model: %s", model_path)
    LOGGER.info("Images found: %d", len(image_paths))

    if not image_paths:
        return

    run_inference(
        model=model,
        image_paths=image_paths,
        dict_encoding=dict_encoding,
        device=device,
        input_h=int(cfg.INPUT.SIZE_CRNN[0]),
        input_w=int(cfg.INPUT.SIZE_CRNN[1]),
        pixel_mean=float(cfg.INPUT.PIXEL_MEAN),
        pixel_std=float(cfg.INPUT.PIXEL_STD),
        invert_threshold=float(args.invert_threshold),
        batch_size=int(args.batch_size),
        use_amp=bool(args.use_amp),
        warmup_iters=max(int(args.warmup_iters), 0),
    )


if __name__ == "__main__":
    main()
