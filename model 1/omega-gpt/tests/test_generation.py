"""Unit tests for Omega GPT-1 Generation and Sampling."""

import unittest
import torch
from config.model_config import ModelConfig
from generation.sampling import sample_next_token
from model.gpt import OmegaGPT


class TestOmegaGeneration(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=200, context_length=32, d_model=64, num_layers=2, num_heads=2, d_ff=128
        )
        self.model = OmegaGPT(self.config)
        self.model.eval()

    def test_sampling_shapes(self):
        logits = torch.randn(1, self.config.vocab_size)
        token = sample_next_token(logits, temperature=0.8, top_k=20, top_p=0.9)
        self.assertEqual(token.shape, (1, 1))
        self.assertTrue(0 <= token.item() < self.config.vocab_size)

    def test_greedy_sampling(self):
        logits = torch.zeros(1, self.config.vocab_size)
        logits[0, 42] = 100.0  # Dominant logit
        token = sample_next_token(logits, temperature=0.0)
        self.assertEqual(token.item(), 42)

    def test_model_generate(self):
        prompt_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        output_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
        )
        # Length should be original (5) + new (10) = 15
        self.assertEqual(output_ids.shape, (1, 15))


if __name__ == "__main__":
    unittest.main()

