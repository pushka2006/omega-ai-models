"""Unit tests for Omega GPT-1 Training Components."""

import os
import shutil
import tempfile
import unittest
import torch
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from model.gpt import OmegaGPT
from training.checkpoint import CheckpointManager
from training.dataset import OmegaDataset
from training.optimizer import build_optimizer
from training.scheduler import CosineWarmupScheduler


class TestOmegaTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = ModelConfig(
            vocab_size=200, context_length=16, d_model=64, num_layers=2, num_heads=2, d_ff=128
        )
        self.train_cfg = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_steps=10,
            warmup_steps=2,
            decay_steps=10,
            checkpoint_dir=os.path.join(self.temp_dir, "checkpoints"),
        )
        self.model = OmegaGPT(self.model_cfg)
        self.optimizer = build_optimizer(self.model, self.train_cfg)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_item_shape(self):
        tokens = torch.randint(0, 200, (100,))
        dataset = OmegaDataset(tokens, context_length=16)
        x, y = dataset[0]
        self.assertEqual(x.shape, (16,))
        self.assertEqual(y.shape, (16,))
        # Target y is shifted by 1 relative to x
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_backpropagation_and_step(self):
        x = torch.randint(0, 200, (2, 16))
        y = torch.randint(0, 200, (2, 16))

        self.model.train()
        logits, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in self.model.parameters())
        self.assertTrue(has_grad)

        self.optimizer.step()

    def test_scheduler_warmup_and_decay(self):
        scheduler = CosineWarmupScheduler(self.optimizer, self.train_cfg)
        lr_0 = scheduler.step(0)
        lr_1 = scheduler.step(1)
        # In warmup, lr increases
        self.assertTrue(lr_1 > lr_0)

    def test_checkpoint_save_and_load(self):
        ckpt_mgr = CheckpointManager(self.train_cfg.checkpoint_dir)
        save_path = ckpt_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            step=5,
            val_loss=2.5,
            model_config=self.model_cfg,
            training_config=self.train_cfg,
        )
        self.assertTrue(os.path.exists(save_path))

        new_model = OmegaGPT(self.model_cfg)
        new_optimizer = build_optimizer(new_model, self.train_cfg)
        ckpt = ckpt_mgr.load(save_path, new_model, new_optimizer)

        self.assertEqual(ckpt["step"], 5)
        self.assertAlmostEqual(ckpt["val_loss"], 2.5)


if __name__ == "__main__":
    unittest.main()

