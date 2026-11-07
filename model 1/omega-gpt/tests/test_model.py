"""Unit tests for Omega GPT-1 Architecture."""

import unittest
import torch
from config.model_config import ModelConfig
from model.attention import CausalSelfAttention
from model.embeddings import OmegaEmbeddings
from model.gpt import OmegaGPT
from model.model_utils import count_parameters


class TestOmegaModel(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=1000,
            context_length=64,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            dropout=0.0,
            tie_word_embeddings=True,
        )
        self.model = OmegaGPT(self.config)

    def test_embeddings_shape(self):
        emb = OmegaEmbeddings(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 32))
        out = emb(x)
        self.assertEqual(out.shape, (2, 32, self.config.d_model))

    def test_attention_shape(self):
        attn = CausalSelfAttention(self.config)
        x = torch.randn(2, 32, self.config.d_model)
        out = attn(x)
        self.assertEqual(out.shape, (2, 32, self.config.d_model))

    def test_forward_logits_and_loss(self):
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        logits, loss = self.model(input_ids, targets=targets)

        # Check shape: (B, T, vocab_size)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() > 0.0)

    def test_parameter_count(self):
        total, trainable = count_parameters(self.model)
        self.assertTrue(total > 10_000)
        self.assertEqual(total, trainable)


if __name__ == "__main__":
    unittest.main()

