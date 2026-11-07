"""Automated unit and integration test suite for OMEGA NEXUS MODEL."""

import unittest
import torch

from core.fusion_layer import MultimodalProjector, CrossAttentionFusion, NexusFusionNetwork
from core.orchestrator import OmegaNexus


class TestOmegaNexus(unittest.TestCase):
    def setUp(self):
        self.nexus = OmegaNexus()

    def test_01_multimodal_projector(self):
        """Verify feature and text projection dimensions."""
        proj = MultimodalProjector(input_dim=16, latent_dim=256)
        x = torch.randn(4, 16)
        out = proj(x)
        self.assertEqual(out.shape, (4, 256))

    def test_02_cross_attention_fusion(self):
        """Verify cross-attention fusion between query and key_value."""
        cross_attn = CrossAttentionFusion(latent_dim=256, num_heads=4)
        query = torch.randn(2, 8, 256)
        kv = torch.randn(2, 4, 256)
        fused = cross_attn(query, kv)
        self.assertEqual(fused.shape, (2, 8, 256))

    def test_03_nexus_fusion_network(self):
        """Verify end-to-end multimodal network forward pass and logits."""
        net = NexusFusionNetwork(text_dim=256, feature_dim=16, latent_dim=256, num_classes=5)
        text_emb = torch.randn(2, 6, 256)
        features = torch.randn(2, 16)
        fused, logits = net(text_emb, features)
        self.assertEqual(fused.shape, (2, 6, 256))
        self.assertEqual(logits.shape, (2, 5))

    def test_04_system_status(self):
        """Verify telemetry reports all 4 models in ecosystem."""
        status = self.nexus.get_system_status()
        self.assertIn("models", status)
        self.assertIn("model_1_omega_gpt1", status["models"])
        self.assertIn("model_2_omega_assistant", status["models"])
        self.assertIn("model_3_neural_engine", status["models"])
        self.assertIn("model_4_omega_nexus", status["models"])

    def test_05_multimodal_dispatch(self):
        """Verify dispatch with multimodal feature payload."""
        features = torch.randn(1, 16)
        res = self.nexus.dispatch("Analyze sensory signal", features=features, mode="multimodal")
        self.assertEqual(res["source"], "model_4_nexus_fusion")
        self.assertIn("fused_shape", res)
        self.assertIn("predicted_class", res)


if __name__ == "__main__":
    unittest.main()

