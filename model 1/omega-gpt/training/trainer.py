"""Omega GPT-1 Training Engine."""

import math
import os
import time
from typing import Dict, List, Optional
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from model.gpt import OmegaGPT
from model.model_utils import count_parameters, format_parameter_count
from tokenizer.tokenizer_utils import OmegaTokenizer
from .checkpoint import CheckpointManager
from .dataset import OmegaDataset
from .evaluate import evaluate_model
from .optimizer import build_optimizer
from .scheduler import CosineWarmupScheduler


class OmegaTrainer:
    """Production training engine for Omega GPT-1."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer: Optional[OmegaTokenizer] = None,
    ):
        self.model_config = model_config
        self.config = training_config

        # Device determination
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        print(f"[Trainer] Selected compute device: {self.device} (type: {self.device_type})")

        # Set random seeds
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # Output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        os.makedirs(self.config.graph_dir, exist_ok=True)

        # Initialize Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = OmegaTokenizer.load(self.config.tokenizer_dir)
            except Exception:
                self.tokenizer = None

        # Build Model from scratch (Random Initialization)
        print("[Trainer] Initializing Omega GPT-1 model architecture with random weights...")
        self.model = OmegaGPT(self.model_config).to(self.device)
        self.model.print_summary()

        # Optimizer and Scheduler
        self.optimizer = build_optimizer(self.model, self.config, device_type=self.device_type)
        self.scheduler = CosineWarmupScheduler(self.optimizer, self.config)

        # Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        # Mixed Precision Scaler
        self.use_amp = self.config.mixed_precision and (self.device_type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # TensorBoard Writer
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        except Exception:
            pass

        # Metrics history
        self.history = {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "perplexity": [],
            "tokens_processed": [],
        }

    def train(
        self,
        train_tokens_path: str = "data/processed/train_tokens.pt",
        val_tokens_path: str = "data/processed/val_tokens.pt",
        resume: bool = False,
    ):
        """Execute the primary training loop."""
        if not os.path.exists(train_tokens_path) or not os.path.exists(val_tokens_path):
            raise FileNotFoundError(
                f"Token datasets not found at {train_tokens_path} or {val_tokens_path}. "
                "Run 'python main.py prepare-data' first."
            )

        train_tensor = torch.load(train_tokens_path, weights_only=True)
        val_tensor = torch.load(val_tokens_path, weights_only=True)

        train_dataset = OmegaDataset(train_tensor, self.model_config.context_length)
        val_dataset = OmegaDataset(val_tensor, self.model_config.context_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(self.device_type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        # Checkpoint Resume
        start_step = 0
        if resume:
            latest_path = self.checkpoint_manager.get_latest_checkpoint_path()
            if latest_path:
                ckpt = self.checkpoint_manager.load(
                    latest_path, self.model, self.optimizer, device=self.device
                )
                start_step = ckpt.get("step", 0)
                print(f"[Trainer] Resuming training from step {start_step}")
            else:
                print("[Trainer] No checkpoint found to resume from. Starting from step 0.")

        print("=" * 60)
        print("STARTING OMEGA GPT-1 TRAINING")
        print(f"Max steps           : {self.config.max_steps}")
        print(f"Batch size          : {self.config.batch_size}")
        print(f"Context length      : {self.model_config.context_length}")
        print(f"Learning rate       : {self.config.learning_rate} -> {self.config.min_learning_rate}")
        print(f"Mixed precision AMP : {self.use_amp}")
        print("=" * 60)

        self.model.train()
        train_iter = iter(train_loader)
        total_tokens_processed = start_step * self.config.batch_size * self.model_config.context_length
        start_time = time.time()
        last_log_time = time.time()

        for step in range(start_step, self.config.max_steps):
            t0 = time.time()

            # Fetch next batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Update learning rate
            lr = self.scheduler.step(step)

            # Forward pass under mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits, loss = self.model(x, y)
                loss_scaled = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss_scaled).backward()

            # Optimizer step with gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = 0.0

            step_tokens = self.config.batch_size * self.model_config.context_length
            total_tokens_processed += step_tokens
            step_duration = time.time() - t0
            tok_per_sec = step_tokens / max(step_duration, 1e-5)

            # Periodic logging
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                cur_loss = loss.item()
                elapsed = time.time() - start_time
                print(
                    f"Step {step:5d}/{self.config.max_steps} | "
                    f"Loss: {cur_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {tok_per_sec:,.0f} tok/s | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                self.history["steps"].append(step)
                self.history["train_loss"].append(cur_loss)
                self.history["learning_rate"].append(lr)
                self.history["tokens_processed"].append(total_tokens_processed)

                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", cur_loss, step)
                    self.writer.add_scalar("Train/LearningRate", lr, step)
                    self.writer.add_scalar("Train/Throughput_TokSec", tok_per_sec, step)

            # Periodic Evaluation & Validation
            if (
                step > 0
                and step % self.config.eval_interval == 0
                or step == self.config.max_steps - 1
            ):
                eval_metrics = evaluate_model(
                    self.model, val_loader, max_eval_steps=self.config.eval_steps, device=self.device
                )
                val_loss = eval_metrics["val_loss"]
                ppl = eval_metrics["perplexity"]

                print("-" * 60)
                print(f"[EVALUATION @ Step {step}] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
                print("-" * 60)

                self.history["val_loss"].append((step, val_loss))
                self.history["perplexity"].append((step, ppl))

                if self.writer is not None:
                    self.writer.add_scalar("Val/Loss", val_loss, step)
                    self.writer.add_scalar("Val/Perplexity", ppl, step)

                # Save checkpoint
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    val_loss=val_loss,
                    model_config=self.model_config,
                    training_config=self.config,
                )

                # Save training metrics plots
                self.save_training_plots()

            # Periodic Generation Sample
            if (
                step > 0
                and step % self.config.generate_interval == 0
                or step == self.config.max_steps - 1
            ):
                self.generate_training_sample(step)

        total_time = time.time() - start_time
        print("=" * 60)
        print(f"TRAINING COMPLETED SUCCESSFULLY in {total_time:.2f}s!")
        print(f"Total tokens processed: {total_tokens_processed:,}")
        print("=" * 60)

        # Final plots and cleanup
        self.save_training_plots()
        if self.writer is not None:
            self.writer.close()

    def generate_training_sample(self, step: int):
        """Generate a brief text sample during training to observe learning progress."""
        if self.tokenizer is None:
            return

        self.model.eval()
        prompt = "The future of artificial intelligence"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=40,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                eos_id=self.tokenizer.eos_id,
            )

        text = self.tokenizer.decode(out_ids[0].tolist())
        sample_path = os.path.join(self.config.sample_dir, f"sample_step_{step:05d}.txt")

        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(f"Step: {step}\nPrompt: {prompt}\nGenerated:\n{text}\n")

        print(f"[Sample @ Step {step}] '{text[:120]}...'")
        self.model.train()

    def save_training_plots(self):
        """Generate and save training loss, validation loss, perplexity, and learning rate curves."""
        if not self.history["steps"]:
            return

        try:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Training & Validation Loss
            axs[0, 0].plot(self.history["steps"], self.history["train_loss"], label="Train Loss", color="royalblue")
            if self.history["val_loss"]:
                v_steps, v_losses = zip(*self.history["val_loss"])
                axs[0, 0].plot(v_steps, v_losses, label="Validation Loss", color="crimson", marker="o")
            axs[0, 0].set_title("Cross-Entropy Loss")
            axs[0, 0].set_xlabel("Steps")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].grid(True, linestyle="--", alpha=0.6)
            axs[0, 0].legend()

            # 2. Validation Perplexity
            if self.history["perplexity"]:
                p_steps, p_vals = zip(*self.history["perplexity"])
                axs[0, 1].plot(p_steps, p_vals, label="Validation Perplexity", color="darkgreen", marker="s")
                axs[0, 1].set_title("Validation Perplexity")
                axs[0, 1].set_xlabel("Steps")
                axs[0, 1].set_ylabel("Perplexity (e^Loss)")
                axs[0, 1].grid(True, linestyle="--", alpha=0.6)
                axs[0, 1].legend()

            # 3. Learning Rate Schedule
            axs[1, 0].plot(self.history["steps"], self.history["learning_rate"], label="Learning Rate", color="darkorange")
            axs[1, 0].set_title("Cosine Warmup Schedule")
            axs[1, 0].set_xlabel("Steps")
            axs[1, 0].set_ylabel("LR")
            axs[1, 0].grid(True, linestyle="--", alpha=0.6)
            axs[1, 0].legend()

            # 4. Total Tokens Processed
            axs[1, 1].plot(self.history["steps"], self.history["tokens_processed"], label="Tokens", color="purple")
            axs[1, 1].set_title("Cumulative Tokens Processed")
            axs[1, 1].set_xlabel("Steps")
            axs[1, 1].set_ylabel("Tokens")
            axs[1, 1].grid(True, linestyle="--", alpha=0.6)
            axs[1, 1].legend()

            plt.tight_layout()
            plot_path = os.path.join(self.config.graph_dir, "training_metrics.png")
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[Trainer] Plot saving notice: {e}")

