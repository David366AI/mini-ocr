# mini-ocr

Mini OCR is a compact CRNN-based project for recognizing short numeric text from grayscale images.

## What Problem This Project Solves

This project is designed for OCR tasks where labels are embedded in filenames and the target text is mainly numbers and number-like strings (for example `771.99.26`, `-6`, or `12,345.67`).

It helps you:

- generate synthetic OCR datasets quickly,
- train a lightweight OCR model with CTC loss,
- run fast PyTorch inference from `.pt` checkpoints.

## Typical Application Scenarios

- printed numeric strings on receipts and forms,
- industrial meters and machine panels,
- serial/batch number extraction,
- financial number snapshots in controlled layouts.

## Post-Training and Extension Support

The project supports continued training (fine-tuning) from a base checkpoint and easy extension:

- continue training from a pretrained `.pt` model via `SOLVER.PRETRAINED_MODEL`,
- switch/expand character dictionary by changing `DATASETS.CHAR_DICT`,
- generate more synthetic data by adding fonts in `pseudo/en` and `pseudo/cn`,
- add more backgrounds in `pseudo/background`,
- tune augmentation in `ocr/utils/enhance.py`.

## Installation

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

If you upload `print_digital_base.pt` to the server (for example under `./data/models/print_digital_base.pt`), run:

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
