"""Dataset implementation for mini OCR training.

Author: David
"""

import logging
import os
from typing import Optional

import cv2
from torch.utils.data import Dataset

from ocr.config.dict_encoding import DictEncoding
from ocr.utils.enhance import enhance_process_image
from ocr.utils.image_utils import recognition_transform, resize_gray_image

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/train.log"),
    ],
)
LOGGER = logging.getLogger(__name__)


class MiniOcrDatabase(Dataset):
    """OCR dataset where labels are embedded in file names.

    Expected naming format: `<prefix>_<label>.jpg` or `.png`.
    """

    def __init__(
        self,
        mode: str,
        cfg,
        dict_encoding: Optional[DictEncoding] = None,
        debug: bool = False,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.cfg = cfg
        self.debug = debug
        self.dict = dict_encoding or DictEncoding(cfg.DATASETS.CHAR_DICT)

        self.image_dir = self._resolve_image_dir()
        self.input_w = int(cfg.INPUT.SIZE_CRNN[1])
        self.input_h = int(cfg.INPUT.SIZE_CRNN[0])
        self.max_seq_len = int(cfg.MODEL.MAX_SEQ_LEN)
        self.mean = float(cfg.INPUT.PIXEL_MEAN)
        self.std = float(cfg.INPUT.PIXEL_STD)

        self.image_name_list, self.label_list = self._collect_samples()

    @staticmethod
    def _normalize_path(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            raise ValueError(f"Dataset path must be a string, got {value!r}")
        return str(value)

    def _resolve_image_dir(self) -> str:
        mode_to_cfg = {
            "train": self.cfg.DATASETS.TRAIN_PATH,
            "val": self.cfg.DATASETS.VAL_PATH,
            "test": self.cfg.DATASETS.TEST_PATH,
        }

        image_dir = self._normalize_path(mode_to_cfg[self.mode])

        if not image_dir:
            raise ValueError(f"No image directory configured for mode: {self.mode}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        return image_dir

    def _collect_samples(self) -> tuple[list[str], list[str]]:
        if self.debug:
            LOGGER.info("Scanning files in %s", self.image_dir)

        image_name_list: list[str] = []
        label_list: list[str] = []

        filenames = sorted(
            name
            for name in os.listdir(self.image_dir)
            if name.lower().endswith((".jpg", ".png"))
        )

        for filename in filenames:
            stem, _ = os.path.splitext(filename)
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue

            image_name_list.append(filename)
            label_list.append(parts[1])

        if self.debug:
            LOGGER.info("Collected %d samples from %s", len(label_list), self.image_dir)

        return image_name_list, label_list

    def __getitem__(self, idx: int):
        image_name = self.image_name_list[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image = enhance_process_image(image)
        image = resize_gray_image(image, self.input_h, self.input_w)

        image_tensor, label_encode, label_len = recognition_transform(
            image,
            self.label_list[idx],
            mean=self.mean,
            std=self.std,
            alphabet_dict=self.dict.dict_mapping,
            max_seq_len=self.max_seq_len,
        )
        return image_tensor, label_encode, label_len

    def __len__(self) -> int:
        return len(self.image_name_list)
