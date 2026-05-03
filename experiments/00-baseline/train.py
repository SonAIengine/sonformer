"""
00-baseline training script.

Self-contained: clone this folder anywhere, install requirements.txt,
`bash run.sh` and you have a baseline run.

Pipeline:
    1. Load config (config.yaml)
    2. Get tokenizer (auto-train BPE on chosen dataset if not cached)
    3. Build streaming dataset → DataLoader
    4. Build model + optimizer + LR schedule
    5. Train with periodic eval, log to results/train_log.csv
    6. Save best checkpoint to results/best.pt
    7. Final eval + generation samples → results/eval.json, samples.txt
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# repo root을 path에 추가 → shared/ import 가능
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.datasets import load_dataset  # noqa: E402
from shared.eval import generate_samples, perplexity  # noqa: E402
from shared.logging import CSVLogger, WandbLogger, save_json, save_samples  # noqa: E402
from shared.tokenizers import get_tokenizer  # noqa: E402

from model import Config, Sonformer  # noqa: E402  (local)


# ───────────────────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """텍스트 리스트를 받아 토큰화 후 (x, y) 페어 반환."""

    def __init__(self, texts, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 미리 전부 토큰화해 메모리에 올림 (TinyStories 정도면 OK)
        self.samples = []
        for t in texts:
            ids = tokenizer.encode(t).ids
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) >= 2:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    L = max(len(x) for x in xs)
    px = torch.full((len(xs), L), pad_id, dtype=torch.long)
    py = torch.full((len(ys), L), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px[i, :len(x)] = x
        py[i, :len(y)] = y
    return px, py


# ───────────────────────────────────────────────────────────────────────────
# LR schedule
# ───────────────────────────────────────────────────────────────────────────

def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * step / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg["min_lr"] + (cfg["learning_rate"] - cfg["min_lr"]) * coeff


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent
    cfg = yaml.safe_load((here / "config.yaml").read_text())

    # seed
    seed = cfg["train"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    # device
    dev = cfg["train"]["device"]
    if dev == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev)
    print(f"[device] {device}")

    # results dir
    results_dir = here / "results"
    results_dir.mkdir(exist_ok=True)

    # ── data + tokenizer ──────────────────────────────────────────────────
    ds_name = cfg["data"]["dataset"]
    vocab = cfg["data"]["vocab_size"]
    max_len = cfg["model"]["max_seq_len"]

    print(f"[data] loading {ds_name} ...")
    train_texts = load_dataset(ds_name, split="train", max_samples=cfg["data"]["max_train_samples"])
    val_texts = load_dataset(ds_name, split="validation", max_samples=cfg["data"]["max_eval_samples"])
    print(f"[data] train={len(train_texts):,}  val={len(val_texts):,}")

    tk = get_tokenizer(dataset=ds_name, vocab_size=vocab, train_corpus=train_texts)
    pad_id = tk.token_to_id("<|pad|>") or 0

    print(f"[data] tokenizing ...")
    train_ds = TextDataset(train_texts, tk, max_len)
    val_ds = TextDataset(val_texts, tk, max_len)
    print(f"[data] train tokens kept={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=pad_id),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id=pad_id),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── model ─────────────────────────────────────────────────────────────
    mcfg = Config(
        vocab_size=vocab,
        max_seq_len=max_len,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_hidden=cfg["model"]["ffn_hidden"],
        dropout=cfg["model"]["dropout"],
        pad_id=pad_id,
    )
    model = Sonformer(mcfg).to(device)
    n_params = model.param_count()
    print(f"[model] {n_params:,} params ({n_params/1e6:.2f}M)")

    # ── optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
    )
    use_amp = cfg["train"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── logging ───────────────────────────────────────────────────────────
    csv_log = CSVLogger(
        results_dir / "train_log.csv",
        fieldnames=["step", "lr", "train_loss", "eval_loss", "eval_ppl", "wallclock_s", "vram_mb"],
    )
    wandb_log = WandbLogger(
        project="sonformer",
        run_name=cfg["experiment"]["name"],
        config=cfg,
    )
    save_json(results_dir / "config.json", cfg)
    save_json(results_dir / "model_meta.json", {"n_params": n_params, "param_M": n_params / 1e6})

    # ── train loop ────────────────────────────────────────────────────────
    model.train()
    step = 0
    best_eval = float("inf")
    losses_buf = []
    t0 = time.time()

    print(f"\n{'step':>6} | {'lr':>9} | {'train':>8} | {'eval':>8} | {'ppl':>8} | {'time':>7}")
    print("-" * 60)

    while step < cfg["train"]["max_steps"]:
        for x, y in train_loader:
            if step >= cfg["train"]["max_steps"]:
                break

            x, y = x.to(device), y.to(device)
            lr = lr_at(step, cfg["train"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses_buf.append(loss.item())

            # log
            if step % cfg["train"]["log_interval"] == 0:
                avg = sum(losses_buf[-cfg["train"]["log_interval"]:]) / max(1, len(losses_buf[-cfg["train"]["log_interval"]:]))
                vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:9.6f} | {avg:8.4f} | {'--':>8} | {'--':>8} | {elapsed:6.1f}s")
                csv_log.log(step=step, lr=lr, train_loss=avg, wallclock_s=elapsed, vram_mb=vram)
                wandb_log.log({"train/loss": avg, "lr": lr}, step=step)

            # eval
            if step > 0 and step % cfg["train"]["eval_interval"] == 0:
                eval_res = perplexity(model, val_loader, device, max_batches=50, pad_id=pad_id)
                model.train()
                avg_train = sum(losses_buf[-cfg["train"]["eval_interval"]:]) / min(len(losses_buf), cfg["train"]["eval_interval"])
                vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
                elapsed = time.time() - t0
                print(
                    f"{step:6d} | {lr:9.6f} | {avg_train:8.4f} | {eval_res['loss']:8.4f} | {eval_res['ppl']:8.2f} | {elapsed:6.1f}s"
                )
                csv_log.log(
                    step=step, lr=lr, train_loss=avg_train,
                    eval_loss=eval_res["loss"], eval_ppl=eval_res["ppl"],
                    wallclock_s=elapsed, vram_mb=vram,
                )
                wandb_log.log({"eval/loss": eval_res["loss"], "eval/ppl": eval_res["ppl"]}, step=step)

                if eval_res["loss"] < best_eval:
                    best_eval = eval_res["loss"]
                    torch.save(
                        {"step": step, "model_state_dict": model.state_dict(), "config": vars(mcfg), "eval_loss": eval_res["loss"]},
                        results_dir / "best.pt",
                    )
                    print(f"   → new best (eval_loss={eval_res['loss']:.4f})")

            step += 1

    # ── final eval + samples ──────────────────────────────────────────────
    print("\n[final] eval ...")
    final_eval = perplexity(model, val_loader, device, max_batches=200, pad_id=pad_id)
    print(f"[final] eval_loss={final_eval['loss']:.4f}  ppl={final_eval['ppl']:.2f}")

    print("[final] generation samples ...")
    samples = generate_samples(
        model, tk, cfg["generation"]["prompts"], device,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
        top_k=cfg["generation"]["top_k"],
    )

    save_json(results_dir / "eval.json", {
        "final": final_eval,
        "best_eval_loss": best_eval,
        "n_params": n_params,
        "total_steps": step,
        "wallclock_s": time.time() - t0,
    })
    save_samples(results_dir / "samples.txt", samples)

    csv_log.close()
    wandb_log.finish()
    print(f"\n[done] results in {results_dir}")


if __name__ == "__main__":
    main()
