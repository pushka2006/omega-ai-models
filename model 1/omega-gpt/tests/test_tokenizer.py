"""Unit tests for Omega GPT-1 Tokenizer."""

import os
import shutil
import tempfile
import unittest

from tokenizer.tokenizer_utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, OmegaTokenizer
from tokenizer.train_tokenizer import train_bpe_tokenizer


class TestOmegaTokenizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_text_file = os.path.join(self.test_dir, "corpus.txt")
        with open(self.sample_text_file, "w", encoding="utf-8") as f:
            f.write(
                "Artificial intelligence and deep learning transformer language models are trained from scratch. "
                "The attention mechanism allows neural networks to capture long range contextual dependencies."
            )
        self.save_dir = os.path.join(self.test_dir, "saved_tok")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_save_load(self):
        tok = train_bpe_tokenizer([self.sample_text_file], vocab_size=300, save_dir=self.save_dir)
        self.assertTrue(tok.get_vocab_size() > 10)

        # Reload from disk
        loaded_tok = OmegaTokenizer.load(self.save_dir)
        self.assertEqual(tok.get_vocab_size(), loaded_tok.get_vocab_size())

    def test_encode_decode(self):
        tok = train_bpe_tokenizer([self.sample_text_file], vocab_size=300, save_dir=self.save_dir)
        text = "deep learning transformer"
        encoded = tok.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(encoded) > 0)

        decoded = tok.decode(encoded)
        self.assertIn("transformer", decoded.lower())

    def test_special_tokens(self):
        tok = train_bpe_tokenizer([self.sample_text_file], vocab_size=300, save_dir=self.save_dir)
        text = "neural networks"
        with_special = tok.encode(text, add_special_tokens=True)
        self.assertEqual(with_special[0], tok.bos_id)
        self.assertEqual(with_special[-1], tok.eos_id)


if __name__ == "__main__":
    unittest.main()

