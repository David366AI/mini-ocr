"""Mini OCR model package.

Author: David
"""

from ocr.model.crnn import MiniCRNN
from ocr.model.minicnn import MiniCNN
from ocr.model.vgg16 import MiniVggCNN

__all__ = ["MiniCRNN", "MiniCNN", "MiniVggCNN"]
