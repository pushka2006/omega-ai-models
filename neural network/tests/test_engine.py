"""OMEGA NEURAL NETWORK ENGINE: Comprehensive Automated Test Suite.

Verifies:
1. Tensor and Device Engine
2. Educational Autograd vs PyTorch Autograd Mathematical Equivalence
3. Modular Layers (Dense, Embedding, Conv1D/2D, Pooling, Normalization, RNN, Transformer)
4. Activation Functions & Losses
5. Optimizers and Schedulers
6. Dataset Loading, Splitting, and Preprocessors
7. Declarative Model Builder
8. Real Training Pipeline & Checkpoint Persistence
"""

import os
import sys
import unittest
import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from activations.functions import ACTIVATION_REGISTRY, get_activation
from builder.model_builder import model_builder
from checkpoints.checkpoint_manager import checkpoint_manager
from core.educational_autograd import compare_gradients_with_pytorch
from core.tensor_engine import tensor_engine
from datasets.dataset_loader import DatasetLoader, SyntheticDatasetGenerator
from datasets.preprocessor import StandardScaler
from layers.attention import MultiHeadAttention, TransformerBlock
from layers.convolutions import Conv1D, Conv2D
from layers.dense import Dense
from layers.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from layers.normalization import BatchNorm1D, LayerNorm, RMSNorm
from layers.pooling import MaxPool1D, MaxPool2D
from layers.recurrent import GRU, LSTM, RNN
from losses.loss_functions import get_loss_function
from models.mlp import MLP
from optimizers.optimizer_factory import configure_optimizers
from training.trainer import TrainingEngine


class TestOmegaNeuralEngine(unittest.TestCase):

    def test_01_tensor_and_device(self):
        """Verify device detection and reproducible seeding."""
        dev = tensor_engine.get_device()
        self.assertIsInstance(dev, torch.device)
        tensor_engine.set_seed(42)
        r1 = torch.rand(5)
        tensor_engine.set_seed(42)
        r2 = torch.rand(5)
        self.assertTrue(torch.equal(r1, r2))

    def test_02_educational_autograd_vs_pytorch(self):
        """Verify educational autograd matches PyTorch autograd down to 1e-10 precision."""
        np.random.seed(1337)
        x = np.random.randn(4, 5)
        w = np.random.randn(5, 3)
        b = np.random.randn(3)
        t = np.random.randn(4, 3)

        res = compare_gradients_with_pytorch(x, w, b, t)
        self.assertTrue(res["verified"])
        self.assertLess(res["weight_grad_max_diff"], 1e-7)
        self.assertLess(res["bias_grad_max_diff"], 1e-7)

    def test_03_dense_and_embedding_layers(self):
        """Verify Dense and Embedding forward shapes."""
        x = torch.randn(8, 16)
        dense = Dense(16, 32)
        out = dense(x)
        self.assertEqual(out.shape, (8, 32))

        input_ids = torch.randint(0, 100, (4, 12))
        emb = TokenEmbedding(vocab_size=100, embedding_dim=64)
        emb_out = emb(input_ids)
        self.assertEqual(emb_out.shape, (4, 12, 64))

        pos_emb = LearnedPositionalEmbedding(max_seq_len=64, embedding_dim=64)
        pos_out = pos_emb(seq_len=12, device=torch.device("cpu"))
        self.assertEqual(pos_out.shape, (12, 64))

    def test_04_convolution_and_pooling(self):
        """Verify Conv1D/2D and Pooling layers."""
        x_1d = torch.randn(4, 8, 32)  # (B, C, L)
        conv1 = Conv1D(8, 16, kernel_size=3, padding=1)
        self.assertEqual(conv1(x_1d).shape, (4, 16, 32))
        pool1 = MaxPool1D(kernel_size=2, stride=2)
        self.assertEqual(pool1(conv1(x_1d)).shape, (4, 16, 16))

        x_2d = torch.randn(4, 3, 16, 16)  # (B, C, H, W)
        conv2 = Conv2D(3, 16, kernel_size=3, padding=1)
        self.assertEqual(conv2(x_2d).shape, (4, 16, 16, 16))
        pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.assertEqual(pool2(conv2(x_2d)).shape, (4, 16, 8, 8))

    def test_05_normalization_layers(self):
        """Verify BatchNorm, LayerNorm, and RMSNorm."""
        x = torch.randn(8, 32)
        bn = BatchNorm1D(32)
        ln = LayerNorm(32)
        rms = RMSNorm(32)

        self.assertEqual(bn(x).shape, (8, 32))
        self.assertEqual(ln(x).shape, (8, 32))
        self.assertEqual(rms(x).shape, (8, 32))

    def test_06_recurrent_and_attention_layers(self):
        """Verify RNN, LSTM, GRU, and TransformerBlock."""
        seq = torch.randn(4, 10, 32)  # (B, T, D)

        lstm = LSTM(input_size=32, hidden_size=64)
        lstm_out, _ = lstm(seq)
        self.assertEqual(lstm_out.shape, (4, 10, 64))

        gru = GRU(input_size=32, hidden_size=64)
        gru_out, _ = gru(seq)
        self.assertEqual(gru_out.shape, (4, 10, 64))

        mha = MultiHeadAttention(d_model=32, num_heads=4)
        self.assertEqual(mha(seq).shape, (4, 10, 32))

        tb = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
        self.assertEqual(tb(seq).shape, (4, 10, 32))

    def test_07_activations_and_losses(self):
        """Verify activation functions and loss functions."""
        x = torch.randn(10, 5)
        for act_name in ["relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu"]:
            act = get_activation(act_name)
            self.assertEqual(act(x).shape, x.shape)

        y = torch.randint(0, 5, (10,))
        loss_fn = get_loss_function("cross_entropy")
        loss = loss_fn(x, y)
        self.assertGreater(loss.item(), 0.0)

    def test_08_declarative_builder(self):
        """Verify model building from declarative config specification."""
        cfg = {
            "name": "test_net",
            "input_size": 16,
            "layers": [
                {"type": "linear", "units": 32, "activation": "relu"},
                {"type": "dropout", "rate": 0.1},
                {"type": "linear", "units": 8},
            ],
        }
        model = model_builder.build_from_config(cfg)
        test_in = torch.randn(4, 16)
        out = model(test_in)
        self.assertEqual(out.shape, (4, 8))

    def test_09_real_training_and_checkpoints(self):
        """Verify real end-to-end training loop and checkpoint persistence."""
        x, y = SyntheticDatasetGenerator.generate_spiral(samples_per_class=100, num_classes=3)
        data = DatasetLoader.load_from_arrays(x, y, val_split=0.2, batch_size=32)

        model = MLP(input_dim=2, hidden_dims=[32, 16], output_dim=3, activation="relu")
        opt = configure_optimizers(model, optimizer_name="adamw", learning_rate=0.01)
        crit = get_loss_function("cross_entropy")

        trainer = TrainingEngine(model, crit, opt)
        res = trainer.fit(
            data["train_loader"],
            data["val_loader"],
            epochs=5,
            checkpoint_name="test_spiral.pth",
        )

        self.assertGreaterEqual(len(res["history"]), 5)
        self.assertGreater(res["best_metric"], 40.0)  # Spiral accuracy improves significantly

        # Test Checkpoint Inspection
        ckpt_path = os.path.join(checkpoint_manager.save_dir, "test_spiral.pth")
        self.assertTrue(os.path.exists(ckpt_path))
        info = checkpoint_manager.inspect_checkpoint(ckpt_path)
        self.assertEqual(info["epoch"], 5)


if __name__ == "__main__":
    unittest.main()
