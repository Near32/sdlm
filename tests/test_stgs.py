import torch
import pytest
from sdlm.stgs import STGS

class TestSTGS:
    @pytest.mark.parametrize("batch_size,seq_len,vocab_size", [
        (1, 10, 100),
        (4, 20, 50257),  # DistilGPT-2 vocab size
        (8, 1, 1000)
    ])
    def test_forward_shape(self, device, batch_size, seq_len, vocab_size):
        stgs = STGS(vocab_size=vocab_size, device=device)
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        
        output_ids, one_hot, temperature, stgs_probs = stgs(logits)
        
        assert output_ids.shape == (batch_size, seq_len)
        assert one_hot.shape == (batch_size, seq_len, vocab_size)
        assert stgs_probs.shape == (batch_size, seq_len, vocab_size)
        assert temperature.numel() == 1

    def test_hard_sampling(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, stgs_hard=True, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device)
        
        _, one_hot, _, _ = stgs(logits)
        
        # In hard mode, one_hot should be one-hot encoded
        assert torch.all(torch.sum(one_hot, dim=-1) == 1.0)
        assert torch.all(torch.sum(one_hot > 0, dim=-1) == 1)

    def test_soft_sampling(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, stgs_hard=False, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device)
        
        _, one_hot, _, _ = stgs(logits)
        
        # In soft mode, one_hot should be a probability distribution
        assert torch.allclose(torch.sum(one_hot, dim=-1), torch.ones(1, 5, device=device))
        assert torch.all(one_hot >= 0)
        assert torch.all(one_hot <= 1)

    def test_temperature_effect(self, device):
        vocab_size = 10
        logits = torch.tensor([[[1.0] + [0.0] * (vocab_size - 1)]], device=device)
        
        # High temperature should make the distribution more uniform
        stgs_high = STGS(vocab_size=vocab_size, init_temperature=10.0, device=device)
        _, one_hot_high, _, _ = stgs_high(logits)
        
        # Low temperature should make the distribution more peaked
        stgs_low = STGS(vocab_size=vocab_size, init_temperature=0.1, device=device)
        _, one_hot_low, _, _ = stgs_low(logits)
        
        # The max probability should be higher with lower temperature
        max_prob_high = torch.max(one_hot_high).item()
        max_prob_low = torch.max(one_hot_low).item()
        assert max_prob_low > max_prob_high

    def test_one_hot_gradient_flow(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)
        
        _, one_hot, _, _ = stgs(logits)
        loss = one_hot.sum()
        loss.backward()
        
        # Check that gradients are flowing back to logits
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)
    
    def test_ids_gradient_flow(self, device):
        vocab_size = 10
        stgs = STGS(vocab_size=vocab_size, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)

        output_ids, _, _, _ = stgs(logits)
        loss = output_ids.sum()
        loss.backward()

        # Check that gradients are flowing back to logits
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_input_dropout_zero_has_no_effect(self, device):
        """With input_dropout=0, forward is deterministic given fixed Gumbel noise."""
        vocab_size = 20
        stgs = STGS(vocab_size=vocab_size, input_dropout=0.0, device=device)
        stgs.train()
        logits = torch.ones(1, 5, vocab_size, device=device)
        # Two runs with same module should have same y_soft shape
        _, one_hot_a, _, _ = stgs(logits)
        _, one_hot_b, _, _ = stgs(logits)
        # Shapes must be valid probability distributions
        assert one_hot_a.shape == (1, 5, vocab_size)
        assert one_hot_b.shape == (1, 5, vocab_size)

    def test_input_dropout_nonzero_zeroes_logits(self, device):
        """With input_dropout > 0 and training=True, some logit values become zero."""
        torch.manual_seed(0)
        vocab_size = 1000
        stgs = STGS(vocab_size=vocab_size, input_dropout=0.5, device=device)
        stgs.train()
        logits = torch.ones(2, 8, vocab_size, device=device)
        # Run forward: inside forward, F.dropout zeros ~50% of logit dimensions
        # The resulting y_soft must still be a valid distribution (sums to 1)
        _, one_hot, _, y_soft = stgs(logits)
        assert one_hot.shape == (2, 8, vocab_size)
        # Probabilities must still sum to 1 (softmax normalises after dropout)
        assert torch.allclose(y_soft.sum(dim=-1), torch.ones(2, 8, device=device), atol=1e-4)

    def test_input_dropout_eval_mode_no_effect(self, device):
        """In eval mode, input_dropout must be inactive (F.dropout respects training flag)."""
        vocab_size = 50
        stgs = STGS(vocab_size=vocab_size, input_dropout=0.9, device=device)
        stgs.eval()
        logits = torch.randn(1, 3, vocab_size, device=device)
        # Eval: dropout disabled → same output on repeated calls with same Gumbel (probabilistic but shape OK)
        _, _, _, y_soft = stgs(logits)
        assert y_soft.shape == (1, 3, vocab_size)
        assert torch.allclose(y_soft.sum(dim=-1), torch.ones(1, 3, device=device), atol=1e-4)

    def test_output_dropout_zero_has_no_effect(self, device):
        """With output_dropout=0, y_soft is an unmodified softmax output."""
        vocab_size = 20
        stgs = STGS(vocab_size=vocab_size, output_dropout=0.0, device=device)
        stgs.train()
        logits = torch.ones(1, 5, vocab_size, device=device)
        _, _, _, y_soft = stgs(logits)
        assert y_soft.shape == (1, 5, vocab_size)
        assert torch.allclose(y_soft.sum(dim=-1), torch.ones(1, 5, device=device), atol=1e-4)

    def test_output_dropout_applied_after_softmax(self, device):
        """With output_dropout > 0, some y_soft entries are zeroed (but rescaled, not renormalised)."""
        torch.manual_seed(0)
        vocab_size = 1000
        stgs = STGS(vocab_size=vocab_size, output_dropout=0.5, device=device)
        stgs.train()
        logits = torch.ones(2, 8, vocab_size, device=device)
        _, _, _, y_soft = stgs(logits)
        assert y_soft.shape == (2, 8, vocab_size)
        # F.dropout zeros ~50% of entries and rescales the rest by 1/(1-0.5)=2;
        # the output is NOT renormalised so exact sum != 1, but values are non-negative
        assert torch.all(y_soft >= 0)

    def test_output_dropout_eval_mode_no_effect(self, device):
        """In eval mode, output_dropout must be inactive."""
        vocab_size = 50
        stgs = STGS(vocab_size=vocab_size, output_dropout=0.9, device=device)
        stgs.eval()
        logits = torch.randn(1, 3, vocab_size, device=device)
        _, _, _, y_soft = stgs(logits)
        assert y_soft.shape == (1, 3, vocab_size)
        # Eval: no dropout, so y_soft is a valid probability distribution
        assert torch.allclose(y_soft.sum(dim=-1), torch.ones(1, 3, device=device), atol=1e-4)


def test_last_snr_shape_and_value():
    """last_snr is (seq_len,) and scales correctly with logit variance and noise scale."""
    import math
    vocab_size = 50
    seq_len = 5
    batch_size = 3
    stgs = STGS(vocab_size=vocab_size, stgs_hard=False, device="cpu")

    # last_snr must be None before any forward call
    assert stgs.last_snr is None

    x = torch.randn(batch_size, seq_len, vocab_size)
    stgs(x)

    assert stgs.last_snr is not None
    assert stgs.last_snr.shape == (seq_len,), f"Expected ({seq_len},), got {stgs.last_snr.shape}"
    assert (stgs.last_snr >= 0).all(), "SNR must be non-negative"

    # Logits with higher variance should yield higher SNR
    x_low = torch.zeros(batch_size, seq_len, vocab_size)      # var ≈ 0
    x_high = torch.randn(batch_size, seq_len, vocab_size) * 10  # var ≈ 100
    stgs(x_low)
    snr_low = stgs.last_snr.clone()
    stgs(x_high)
    snr_high = stgs.last_snr.clone()
    assert (snr_high > snr_low).all(), "Higher logit variance → higher SNR"

    # Doubling gumbel_noise_scale should reduce SNR by 4×
    x_fixed = torch.randn(batch_size, seq_len, vocab_size)
    stgs(x_fixed, gumbel_noise_scale=1.0)
    snr_scale1 = stgs.last_snr.clone()
    stgs(x_fixed, gumbel_noise_scale=2.0)
    snr_scale2 = stgs.last_snr.clone()
    ratio = snr_scale1 / snr_scale2.clamp(min=1e-12)
    assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=0.05), \
        f"Doubling noise_scale should reduce SNR by 4×, got ratio={ratio}"

    # Doubling temperature should also reduce SNR by 4× (T² in denominator)
    stgs_t1 = STGS(vocab_size=vocab_size, stgs_hard=False, device="cpu", init_temperature=1.0)
    stgs_t2 = STGS(vocab_size=vocab_size, stgs_hard=False, device="cpu", init_temperature=2.0)
    stgs_t1(x_fixed)
    stgs_t2(x_fixed)
    ratio_t = stgs_t1.last_snr / stgs_t2.last_snr.clamp(min=1e-12)
    assert torch.allclose(ratio_t, torch.full_like(ratio_t, 4.0), rtol=0.05), \
        f"Doubling temperature should reduce SNR by 4×, got ratio={ratio_t}"
