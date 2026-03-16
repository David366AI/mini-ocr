# mini-ocr

Mini OCR is a lightweight, production-friendly OCR framework built on CRNN + CTC.

`print_digital` is only the default demo task in this repository.  
The core pipeline is generic and can be extended to many OCR scenarios with minimal code changes.

## What Problem This Project Solves

This project gives you a simple end-to-end OCR training and inference stack:

It helps you:

- build labeled OCR datasets quickly (including pseudo data generation),
- train a compact model with stable CTC optimization,
- deploy fast PyTorch inference from `.pt` checkpoints,
- continue training from base models for domain adaptation.

## Typical Application Scenarios

- printed text OCR,
- handwritten OCR,
- meter/panel and instrument reading OCR,
- invoice/receipt key field OCR,
- serial number and code OCR,
- custom vertical-domain OCR with your own alphabet.

## Adapt to Any OCR Task

You can adapt this project from `print_digital` to your own OCR use case by:

- replacing training/validation/test images with your own samples,
- replacing `DATASETS.CHAR_DICT` with your own dictionary file,
- implementing your own dataset parsing logic in `MiniOcrDatabase` when filename labeling is not enough.

The model/training/inference code can stay the same in most cases.

## Post-Training and Extension Support

The project supports continued training (fine-tuning) from a base checkpoint and easy extension:

- continue training from a pretrained `.pt` model via `SOLVER.PRETRAINED_MODEL`,
- switch/expand character dictionary by changing `DATASETS.CHAR_DICT`,
- generate more synthetic data by adding fonts in `pseudo/en` and `pseudo/cn`,
- add more backgrounds in `pseudo/background`,
- tune augmentation in `ocr/utils/enhance.py`.

## Installation

Python requirement:

```text
Python >= 3.10
```

```bash
pip install -r requirements.txt
```

## Data Layout

Expected dataset layout:

```text
data/images/
  train/
  val/
  test/
```

Filename format:

```text
<prefix>_<label>.jpg
```

Example:

```text
1748234847861_-6.jpg
```

## 1) Generate Pseudo Training Data

Use `pseudo/gen_train_images.py`:

```bash
python pseudo/gen_train_images.py -n 8 -o ./data
```

What it does:

- writes raw generated images into `./data/images`,
- shuffles all root `.jpg` files,
- splits them by `9:1`,
- moves files into `./data/images/train` and `./data/images/val`.

Arguments:

- `-n, --num-processes`: number of worker processes (default `1`)
- `-o, --output`: output root path (default `./data`)

## 2) Train

Basic training command:

```bash
python train.py --cfg_file ./config/config_print_digital.yaml
```

Train one epoch for quick test:

```bash
python train.py --cfg_file ./config/config_print_digital.yaml SOLVER.EPOCHS 1
```

Set data loader workers from CLI:

```bash
python train.py --cfg_file ./config/config_print_digital.yaml SOLVER.EPOCHS 100 DATALOADER.NUM_WORKERS 8
```

## 3) Validate / Run Inference

test_4,8383.jpg： ![test sample 1](data/images/test/test_4,8383.jpg)
test_771.99.26.jpg： ![test sample 2](data/images/test/test_771.99.26.jpg)

run:

```bash
python predict.py \
  --cfg-file ./config/config_print_digital.yaml \
  --model-path ./data/models/print_digital_base.pt \
  --image-path ./data/images/test \
  --device cuda \
  --batch-size 32 \
  --warmup-iters 10
```

Output format:

```text
<file_name>\t<predicted_text>\t<latency_ms>

2026-03-15 22:59:21,142 - INFO - Using device: cuda
2026-03-15 22:59:21,143 - INFO - Model: data/models/print_digital_base.pt
2026-03-15 22:59:21,143 - INFO - Images found: 2
2026-03-15 22:59:21,356 - INFO - CUDA warmup finished: 10 iterations (batch=2)
test_771.99.26.jpg       771.99.26       0.262 ms
test_4,8383.jpg          4,8383          0.262 ms
```

## 4) Continue Training from a Base Model

Set this field in your YAML:

```yaml
SOLVER:
  PRETRAINED_MODEL: "./data/models/print_digital_base.pt"
```

Then train again:

```bash
python train.py --cfg_file ./config/config_print_digital.yaml
```

You can still override any config value from CLI, for example:

```bash
python train.py --cfg_file ./config/config_print_digital.yaml SOLVER.EPOCHS 30 SOLVER.LEARNING_RATE 5e-5
```
