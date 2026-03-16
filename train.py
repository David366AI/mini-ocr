"""Training entrypoint for mini OCR.

Author: David
"""

from __future__ import annotations

import argparse
import logging
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ocr.config.dict_encoding import DictEncoding
from ocr.config.loader import load_config
from ocr.dataset.dataset import MiniOcrDatabase
from ocr.model.crnn import MiniCRNN
from ocr.utils.ctc import ctc_decode
from ocr.utils.xer import get_cer, get_wer

LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Mini OCR with CTC loss.")
    parser.add_argument(
        "--cfg_file",
        default="./config/config_print_digital.yaml",
        metavar="FILE",
        help="Path to config file.",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--debug-dataset",
        action="store_true",
        help="Enable dataset debug logging.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using command-line key value pairs.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, dict_encoding: DictEncoding, device: torch.device) -> MiniCRNN:
    """Build the OCR model."""
    model = MiniCRNN(
        charset_size=len(dict_encoding.dicts) + 1,
        backbone=cfg.MODEL.BACKBONE,
        encoder_type=cfg.MODEL.ENCODER_TYPE,
        encoder_input_size=cfg.MODEL.ENCODER_INPUT_SIZE,
        encoder_hidden_size=cfg.MODEL.ENCODER_HIDDEN_SIZE,
        encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        encoder_bidirectional=cfg.MODEL.ENCODER_BIDIRECTIONAL,
        max_seq_len=cfg.MODEL.MAX_SEQ_LEN,
    ).to(device)
    return model


def load_pretrained(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load pretrained weights when configured."""
    if not ckpt_path:
        return

    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {checkpoint_path}")

    LOGGER.info("Loading pretrained weights from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format. Expected a state_dict mapping.")

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def build_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    """Build optimizer from config."""
    optimizer_name = str(cfg.SOLVER.OPTIMIZER).lower()
    learning_rate = float(cfg.SOLVER.LEARNING_RATE)

    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_name == "adam":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER}")


def decode_targets(labels: torch.Tensor, alphabet: str) -> list[str]:
    """Decode padded label tensors into text."""
    decoded: list[str] = []
    labels_np = labels.cpu().numpy()
    alphabet_size = len(alphabet)

    for row in labels_np:
        chars = [
            alphabet[index - 1]
            for index in row
            if index != -1 and 0 < int(index) <= alphabet_size
        ]
        decoded.append("".join(chars))
    return decoded


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CTCLoss,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    print_freq: int,
    clip_grad: float,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm.tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"train {epoch}/{total_epochs}",
    )
    for step, (images, labels, label_lens) in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lens = label_lens.to(device, non_blocking=True).view(-1).long()

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
            batch_size = log_probs.size(1)
            input_lens = torch.full(
                size=(batch_size,),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, labels, input_lens, label_lens) / batch_size

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().item())
        num_batches += 1

        if step % max(print_freq, 1) == 0:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6g}",
            )

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alphabet: str,
    use_amp: bool,
) -> tuple[float, float]:
    """Run validation and return CER/WER in percentage."""
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []

    progress = tqdm.tqdm(dataloader, total=len(dataloader), desc="val", leave=False)
    for images, labels, _ in progress:
        images = images.to(device, non_blocking=True)

        amp_ctx = autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            logits = model(images)

        pred_indices = logits.argmax(dim=2).cpu().numpy()
        hyps.extend(ctc_decode(pred_indices, alphabet_encoding=alphabet))
        refs.extend(decode_targets(labels, alphabet))

    if not refs:
        return 100.0, 100.0

    cer = get_cer(hyps, refs) * 100.0
    wer = get_wer(hyps, refs) * 100.0
    return float(cer), float(wer)


def main() -> None:
    """Main entrypoint."""
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.cfg_file, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    LOGGER.info("Using device: %s", device)

    dict_encoding = DictEncoding(cfg.DATASETS.CHAR_DICT)
    model = build_model(cfg, dict_encoding, device)
    load_pretrained(model, str(cfg.SOLVER.PRETRAINED_MODEL), device)

    train_dataset = MiniOcrDatabase(
        mode="train",
        cfg=cfg,
        dict_encoding=dict_encoding,
        debug=args.debug_dataset,
    )
    val_dataset = MiniOcrDatabase(
        mode="val",
        cfg=cfg,
        dict_encoding=dict_encoding,
        debug=args.debug_dataset,
    )
    LOGGER.info("Training samples: %d", len(train_dataset))
    LOGGER.info("Validation samples: %d", len(val_dataset))

    batch_size = int(cfg.SOLVER.BATCH_SIZE)
    num_workers = int(cfg.DATALOADER.NUM_WORKERS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
        drop_last=False,
    )

    optimizer = build_optimizer(cfg, model)
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=0,
        threshold=0.001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-7,
        eps=1e-7,
    )
    criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    scaler = GradScaler(enabled=use_amp)

    checkpoint_dir = Path(cfg.SOLVER.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / str(cfg.SOLVER.CHECKPOINT_NAME)

    best_cer = float("inf")
    epochs = int(cfg.SOLVER.EPOCHS)
    print_freq = int(cfg.SOLVER.PRINT_FREQ)
    clip_grad = float(cfg.SOLVER.CLIP_GRAD)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=epochs,
            print_freq=print_freq,
            clip_grad=clip_grad,
        )
        cer, wer = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            alphabet=dict_encoding.dicts,
            use_amp=use_amp,
        )
        scheduler.step(cer)

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_cer=%.4f | val_wer=%.4f",
            epoch,
            epochs,
            train_loss,
            cer,
            wer,
        )

        if cer < best_cer:
            LOGGER.info("Saving best checkpoint: %s (CER %.4f -> %.4f)", checkpoint_path, best_cer, cer)
            best_cer = cer
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
