"""Generate synthetic OCR training images.

Author: David
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import random
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.config.dict_encoding import DictEncoding

LOGGER = logging.getLogger("gen_train_images")

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 32
MAX_TEXT_LENGTH = 12
DEFAULT_SAMPLES_PER_FONT = 20000
TRAIN_RATIO = 0.9

ROTATE_VALUES = (-3, -2, -1, 0, 1, 2, 3, 4)
ROTATE_WEIGHTS = (1, 1, 1, 6, 2, 2, 2, 1)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_chars_text(dict_path: Path = PROJECT_ROOT / "config" / "dict_number.txt") -> tuple[str, ...]:
    """Load supported characters from dictionary file via DictEncoding."""
    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    with contextlib.redirect_stdout(io.StringIO()):
        dict_encoding = DictEncoding(str(dict_path))

    chars = tuple(dict_encoding.dicts)
    if not chars:
        raise ValueError(f"Dictionary has no characters: {dict_path}")
    return chars


CHARS_TEXT = load_chars_text()


class FontChecker:
    """Check if a font supports a given unicode character."""

    def __init__(self, font_path: Path):
        self.font = TTFont(str(font_path))
        self.supported_chars = self._extract_supported_chars()

    def _extract_supported_chars(self) -> set[int]:
        supported: set[int] = set()
        for cmap_table in self.font["cmap"].tables:
            supported.update(cmap_table.cmap.keys())
        return supported

    def is_unicode_supported(self, char: str) -> bool:
        return ord(char) in self.supported_chars


def load_font(font_path: Path) -> ImageFont.FreeTypeFont:
    """Load a font with a size tuned to the target image height."""
    target_max_size = int(IMAGE_HEIGHT * 0.92)
    font_size = 24

    while True:
        font = ImageFont.truetype(str(font_path), font_size)
        if font_size >= target_max_size:
            return font
        font_size += 1


def compress_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def add_salt_noise(image: Image.Image, amount: float = 0.005) -> Image.Image:
    """Apply sparse white-pixel noise."""
    arr = np.array(image)
    num_salt = int(np.ceil(amount * arr.shape[0] * arr.shape[1]))

    ys = np.random.randint(0, arr.shape[0], size=num_salt)
    xs = np.random.randint(0, arr.shape[1], size=num_salt)
    arr[ys, xs] = 255
    return Image.fromarray(arr)


def _truncate_text_to_fit(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> tuple[str, tuple[int, int, int, int]]:
    """Truncate text so rendered width fits the image."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]

    if width <= IMAGE_WIDTH * 0.95 or len(text) <= MAX_TEXT_LENGTH:
        return text, bbox

    best_text = ""
    best_bbox = bbox
    for i in range(len(text)):
        candidate = text[: i + 1]
        candidate_bbox = draw.textbbox((0, 0), candidate, font=font)
        candidate_width = candidate_bbox[2] - candidate_bbox[0]
        if candidate_width > IMAGE_WIDTH * 0.95:
            break
        best_text = candidate
        best_bbox = candidate_bbox

    return best_text, best_bbox


def _safe_label_for_filename(label: str) -> str:
    """Convert a label into a filesystem-safe suffix."""
    sanitized = re.sub(r"[^0-9A-Za-z.,\- ]", "", label)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized or "blank"


def generate_text_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    font_checker: FontChecker,
    output_path: Path,
    text_color: tuple[int, int, int] = (0, 0, 0),
    italic: bool = False,
    stroke_width: int = 1,
    background: Image.Image | None = None,
) -> str:
    """Render a text sample and save it to disk.

    Returns the final rendered label (can be truncated), or empty string if invalid.
    """
    if background is not None:
        bg_img = background.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
    else:
        bg_img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")

    draw = ImageDraw.Draw(bg_img)
    filtered_text = "".join(c if font_checker.is_unicode_supported(c) else " " for c in text)
    filtered_text = compress_spaces(filtered_text)

    if not filtered_text:
        return ""

    filtered_text, text_bbox = _truncate_text_to_fit(filtered_text, draw, font)
    if not filtered_text:
        return ""

    text_height = text_bbox[3] - text_bbox[1]
    text_x = random.randint(10, 25) if italic else random.randint(3, 25)
    text_y = (IMAGE_HEIGHT - text_height) // 2 - text_bbox[1] + random.randint(-3, 3)

    if italic:
        temp_img = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text(
            (text_x, text_y),
            filtered_text,
            fill=text_color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=text_color,
        )

        shear_factor = random.choices(ROTATE_VALUES, ROTATE_WEIGHTS, k=1)[0] / 10
        italic_img = temp_img.transform(
            temp_img.size,
            Image.AFFINE,
            (1, shear_factor, 0, 0, 1, 0),
            Image.BICUBIC,
        )
        bg_img.paste(italic_img, (0, 0), italic_img)
    else:
        draw.text(
            (text_x, text_y),
            filtered_text,
            fill=text_color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=text_color,
        )

    if random.random() < 0.4:
        bg_img = add_salt_noise(bg_img, amount=random.uniform(0.001, 0.004))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg_img.save(output_path)
    return filtered_text


def generate_random_digit_strings(
    num_samples: int = 10000,
    chars_text: Sequence[str] = CHARS_TEXT,
    min_len: int = 1,
    max_len: int = 12,
) -> list[str]:
    """Generate random numeric text samples with punctuation."""
    if min_len < 1 or max_len < min_len:
        raise ValueError("Invalid min_len/max_len")

    chars_head = tuple(c for c in chars_text if c not in {",", "."})
    samples: list[str] = []

    for _ in range(num_samples):
        length = random.randint(min_len, max_len)
        if length == 1:
            text = random.choice(chars_head)
        else:
            first = random.choice(chars_head)
            tail = random.choices(chars_text, k=length - 1)
            text = first + "".join(tail)
        samples.append(compress_spaces(text))

    return samples


def discover_fonts() -> list[Path]:
    """Discover all TTF/TTC fonts under pseudo/en and pseudo/cn."""
    font_files: list[Path] = []
    for sub_dir in (SCRIPT_DIR / "en", SCRIPT_DIR / "cn"):
        if not sub_dir.exists():
            continue
        font_files.extend(
            item
            for item in sorted(sub_dir.iterdir())
            if item.suffix.lower() in {".ttf", ".ttc"}
        )
    return font_files


def discover_backgrounds() -> list[Path]:
    """Discover all background images under pseudo/background."""
    background_dir = SCRIPT_DIR / "background"
    if not background_dir.exists():
        return []

    return sorted(
        item
        for item in background_dir.iterdir()
        if item.suffix.lower() in {".jpg", ".png"}
    )


def _load_background_cache(background_files: Sequence[Path]) -> dict[Path, Image.Image]:
    """Load backgrounds into memory and detach file handles."""
    cache: dict[Path, Image.Image] = {}
    for path in background_files:
        with Image.open(path) as img:
            cache[path] = img.convert("RGB")
    return cache


def _move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        timestamp = int(time.time() * 1000)
        dst = dst.with_name(f"{dst.stem}_{timestamp}{dst.suffix}")
    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def process_font(
    index: int,
    font_file: Path,
    images_dir: Path,
    background_files: Sequence[Path],
    samples_per_font: int = DEFAULT_SAMPLES_PER_FONT,
) -> None:
    """Generate images for one font in a worker process."""
    seed = int(time.time() * 1000) ^ os.getpid() ^ index
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    LOGGER.info("Process %s handling font: %s", os.getpid(), font_file.name)

    font = load_font(font_file)
    font_checker = FontChecker(font_file)
    lines = generate_random_digit_strings(num_samples=samples_per_font)

    background_cache = _load_background_cache(background_files)
    bg_weights = [5] + [1] * (len(background_files) - 1)

    for sample_index, line in enumerate(tqdm(lines, desc=f"Rendering {font_file.name}", leave=False)):
        if not line:
            continue

        italic = random.choices([True, False], weights=[1, 3], k=1)[0]
        stroke_width = random.choices([0, 1], weights=[4, 1], k=1)[0]
        color = random.randint(0, 120)

        bg_file = random.choices(background_files, weights=bg_weights, k=1)[0]
        background = background_cache[bg_file]

        label_for_name = _safe_label_for_filename(line)
        unique_prefix = f"{int(time.time() * 1000)}_{os.getpid()}_{index:02d}_{sample_index:05d}"
        filename = images_dir / f"{unique_prefix}_{label_for_name}.jpg"

        rendered = generate_text_image(
            text=line,
            font=font,
            font_checker=font_checker,
            output_path=filename,
            italic=italic,
            stroke_width=stroke_width,
            text_color=(color, color, color),
            background=background,
        )
        if not rendered:
            continue


def split_train_val(images_dir: Path, train_ratio: float = TRAIN_RATIO) -> tuple[int, int, int]:
    """Shuffle and move root JPG files into train/val subdirectories."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    all_jpg_files = [p for p in images_dir.glob("*.jpg")]
    random.shuffle(all_jpg_files)

    split_index = int(len(all_jpg_files) * train_ratio)
    train_files = all_jpg_files[:split_index]
    val_files = all_jpg_files[split_index:]

    train_dir = images_dir / "train"
    val_dir = images_dir / "val"

    for file_path in train_files:
        _move_file(file_path, train_dir / file_path.name)

    for file_path in val_files:
        _move_file(file_path, val_dir / file_path.name)

    return len(all_jpg_files), len(train_files), len(val_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCR training images.")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=1,
        help="Number of worker processes. Default: 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data",
        help="Output root path. Images are written to <output>/images.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.num_processes < 1:
        raise ValueError("-n/--num-processes must be >= 1")

    output_root = Path(args.output).resolve()
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    font_files = discover_fonts()
    if not font_files:
        raise FileNotFoundError("No font files found under pseudo/en or pseudo/cn")

    background_files = discover_backgrounds()
    if not background_files:
        raise FileNotFoundError("No background files found under pseudo/background")

    num_workers = min(args.num_processes, len(font_files))
    tasks = [(i, font_file, images_dir, background_files) for i, font_file in enumerate(font_files)]

    LOGGER.info("Starting generation with %s worker(s)", num_workers)
    if num_workers == 1:
        for task in tasks:
            process_font(*task)
    else:
        with Pool(processes=num_workers) as pool:
            pool.starmap(process_font, tasks)

    total, train_count, val_count = split_train_val(images_dir, train_ratio=TRAIN_RATIO)
    LOGGER.info("Done. Total jpg files: %s", total)
    LOGGER.info("Moved to train: %s", train_count)
    LOGGER.info("Moved to val: %s", val_count)


if __name__ == "__main__":
    main()
