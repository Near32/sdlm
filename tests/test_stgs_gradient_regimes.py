"""
tests/test_stgs_gradient_regimes.py
────────────────────────────────────
Gradient-flow regime tests for the STGS estimator.

These tests document the expected behaviour (and known limitations) of the
Straight-Through Gumbel-Softmax (STGS) estimator across all configuration axes
that appear in the Qwen3-600M LMI run script:

    benchmarking/invertibility/experiment_scripts/lmi/qwen3_600m/
        qwen3_600m_stgs_batch_k1-5-25-run.sh

Outcome labels used in docstrings
──────────────────────────────────
[FLOW]    Gradient exists and is non-zero everywhere – healthy.
[VANISH]  Gradient exists but magnitude is O(1/τ) smaller than at τ=1.
          At τ=100 (run-script default) gradients are ~100× attenuated.
[KILLED]  Gradient is exactly zero or has no computational path (backprop_gate=0).
[PARTIAL] Gradient exists but is zero for a subset of vocab positions.
          Caused by input_dropout (before softmax), NOT output_dropout.
[EDGE]    Edge-case regime with potential numerical distortion.

Key design decisions documented
────────────────────────────────
1.  The existing test_one_hot_gradient_flow uses stgs_hard=False (default),
    exercising the trivial soft path: output_one_hot = y_soft.
    The run script uses stgs_hard=True:
        output_one_hot = y_hard.detach() + gate*(y_soft - y_soft.detach())
    Gradient magnitude scales as 1/τ via the Gumbel-Softmax Jacobian.

2.  Three independent dropout mechanisms (NOT interchangeable):

    dropout (backprop_gate)
        gates the ENTIRE straight-through term for one batch at a time.
        At dropout=1.0 the gate is always 0 → [KILLED].
        With stgs_hard=False the Python if/else takes the discrete branch
          → no computational path → logits.grad is None.
        With stgs_hard=True the gate tensor is 0 → logits.grad is all-zeros.

    input_dropout
        zeroes random logit DIMENSIONS BEFORE Gumbel noise/softmax.
        Dropout mask multiplies the gradient directly at that position.
        → ~p fraction of vocab positions have zero gradient → [PARTIAL].

    output_dropout
        zeroes y_soft entries AFTER softmax, BEFORE hard-token selection.
        The softmax Jacobian COUPLES all vocab positions, so even zeroed
        y_soft positions propagate a non-zero gradient back to the logits.
        → does NOT create zero gradients in logits.grad (unlike input_dropout).

3.  High temperature (τ=100) + embsim-l2 + gumbel_soft probs:
    At τ=100 the y_soft distribution is nearly uniform; the soft embedding
    soft_emb ≈ mean(embedding_matrix), and embsim always picks the same
    "centroid" token regardless of the logits → degenerate token selection.
    Gradient still flows (via straight-through), but the alignment between
    the chosen token and the logit peaks is lost → learning signal degraded.
"""

import pytest
import torch
import torch.nn.functional as F

from sdlm.stgs import STGS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_embeddings(vocab_size: int, embed_dim: int = 64, device="cpu") -> torch.Tensor:
    """Return random unit-normed embedding matrix (V, D)."""
    E = torch.randn(vocab_size, embed_dim, device=device)
    return F.normalize(E, dim=-1)


def _rand_loss(tensor: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Random-weighted dot-product loss for gradient-flow testing.

    Why NOT use tensor.sum():
        sum(softmax(x)) = 1 identically.  Its gradient w.r.t. x is exactly zero
        (the softmax Jacobian rows sum to zero) and may also be exactly zero on
        CUDA due to float32 precision.  This makes `tensor.sum().backward()` an
        unreliable proxy for gradient flow.

    Why this works:
        loss = Σ w_i * y_i  where w_i ~ N(0,1) are random non-uniform weights.
        d(loss)/d(x_j) = Σ_k w_k * d(y_k)/d(x_j) = (1/τ) * y_j * (w_j − Σ_k w_k y_k)
        This is generally non-zero when the w_k are not constant, matching the
        true CE gradient structure used in the actual optimization pipeline.
    """
    g = torch.Generator(device=tensor.device)
    g.manual_seed(seed)
    weights = torch.randn(tensor.shape, generator=g, device=tensor.device, dtype=tensor.dtype)
    return (tensor * weights).sum()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gradient EXISTENCE: hard vs soft, with and without backprop_gate dropout
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientExistence:
    """[FLOW] / [KILLED]: basic existence of gradient across stgs_hard and dropout axes."""

    @pytest.mark.parametrize("stgs_hard", [False, True])
    def test_gradient_exists_hard_and_soft(self, device, stgs_hard):
        """Both soft (trivial) and hard (straight-through) modes must produce gradients.

        NOTE: test_one_hot_gradient_flow only tests stgs_hard=False (the default).
        The run script uses stgs_hard=True.  This test adds that coverage.
        """
        stgs = STGS(vocab_size=20, stgs_hard=stgs_hard, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 20, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None, f"No gradient with stgs_hard={stgs_hard}"
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient with stgs_hard={stgs_hard}"
        )

    @pytest.mark.parametrize("stgs_hard", [False, True])
    def test_backprop_dropout_zero_preserves_gradient(self, device, stgs_hard):
        """dropout=0.0 (default) → backprop_gate=1.0 → [FLOW]."""
        stgs = STGS(vocab_size=20, stgs_hard=stgs_hard, dropout=0.0, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 20, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_backprop_dropout_one_kills_gradient_hard_mode(self, device):
        """stgs_hard=True, dropout=1.0: gate=tensor(0) → gradient is all-zeros. [KILLED]

        The computational graph still EXISTS (y_soft is reachable), but
        multiplying by the zero-gate tensor zeros every gradient element.
        logits.grad is a zero TENSOR, not None.
        """
        torch.manual_seed(0)
        stgs = STGS(vocab_size=20, stgs_hard=True, dropout=1.0, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 20, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None, (
            "Expected a zero-tensor gradient (not None) with stgs_hard=True, dropout=1.0"
        )
        assert torch.all(logits.grad == 0), (
            f"Expected all-zero gradient with dropout=1.0; "
            f"max|grad|={logits.grad.abs().max().item():.3e}"
        )

    def test_backprop_dropout_one_kills_gradient_soft_mode(self, device):
        """stgs_hard=False, dropout=1.0: Python if/else takes the discrete branch. [KILLED]

        When backprop_gate is tensor(0.0), bool(tensor(0.0)) = False, so:
            output_one_hot = F.one_hot(output_ids, ...)   ← no grad_fn
        There is NO computational path back to logits.
        logits.grad is None (or zero if autograd creates a zero tensor).
        """
        torch.manual_seed(0)
        stgs = STGS(vocab_size=20, stgs_hard=False, dropout=1.0, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 20, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)

        # one_hot has no grad_fn; backward may raise or leave grad as None
        if one_hot.requires_grad:
            _rand_loss(one_hot).backward()

        is_dead = logits.grad is None or torch.all(logits.grad == 0)
        assert is_dead, (
            "Expected no gradient with stgs_hard=False, dropout=1.0"
        )

    def test_diff_output_ids_gradient_exists_hard_mode(self, device):
        """The first return value (diff_output_ids) also carries a gradient. [FLOW]

        diff_output_ids = output_ids.detach() + gate*(gathered_one_hot - gathered_one_hot.detach())
        """
        stgs = STGS(vocab_size=20, stgs_hard=True, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 20, device=device, requires_grad=True)

        diff_ids, _, _, _ = stgs(logits)
        _rand_loss(diff_ids).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Gradient MAGNITUDE vs Temperature: 1/τ vanishing
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientMagnitudeVsTemperature:
    """[FLOW] / [VANISH]: gradient magnitude ∝ 1/τ through the Gumbel-Softmax Jacobian.

    The Jacobian element: ∂y_soft_i/∂x_j = y_soft_i(δ_ij − y_soft_j) / τ.
    Every downstream gradient inherits this 1/τ factor.
    At τ=100 (run-script init), gradients are ~100× smaller than at τ=1.
    """

    @pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0, 100.0])
    def test_gradient_exists_at_all_temperatures(self, device, temperature):
        """Gradient must be non-zero for any positive temperature. [FLOW]"""
        stgs = STGS(
            vocab_size=50, stgs_hard=True, init_temperature=temperature, device=device
        )
        stgs.eval()
        logits = torch.randn(2, 5, 50, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient at temperature={temperature}"
        )

    def test_gradient_magnitude_decreases_with_temperature(self, device):
        """Higher temperature → smaller gradient magnitude (∝ 1/τ). [VANISH at τ=100]

        Each 10× temperature increase should reduce the gradient L2 norm by ~10×.
        The run-script learning_rate=1e-1 partially compensates, but callers
        should be aware of the ~100× attenuation at τ=100 vs τ=1.
        """
        vocab_size = 50
        logits_base = torch.randn(2, 5, vocab_size)
        norms = {}

        for tau in [1.0, 10.0, 100.0]:
            stgs = STGS(
                vocab_size=vocab_size, stgs_hard=True,
                init_temperature=tau, device=device
            )
            stgs.eval()
            logits = logits_base.to(device).detach().requires_grad_(True)
            torch.manual_seed(1)           # same Gumbel noise across τ values
            _, one_hot, _, _ = stgs(logits)
            _rand_loss(one_hot).backward()
            norms[tau] = logits.grad.norm().item()

        assert norms[1.0] > norms[10.0], (
            f"Expected norm(τ=1) > norm(τ=10); got {norms[1.0]:.4f} vs {norms[10.0]:.4f}"
        )
        assert norms[10.0] > norms[100.0], (
            f"Expected norm(τ=10) > norm(τ=100); got {norms[10.0]:.4f} vs {norms[100.0]:.4f}"
        )
        # Each decade of τ should roughly halve-to-tenth the gradient norm
        ratio_10_100 = norms[10.0] / norms[100.0]
        assert ratio_10_100 > 3.0, (
            f"Expected gradient ratio(τ=10/τ=100) >> 1 (Jacobian 1/τ scaling); "
            f"got {ratio_10_100:.2f}"
        )

    def test_run_script_temperature_causes_significant_attenuation(self, device):
        """At τ=100 (run-script init) gradient norm is ≥20× smaller than at τ=1. [VANISH]

        Run script: --temperature=100.0 --learnable_temperature=True
        Init value: eff_τ ≈ 100*(1+tanh(~0.5)) ≈ 146 on first forward pass.
        Gradient still FLOWS but the magnitude is severely attenuated.
        The high LR (1e-1) in the run script partially compensates.
        """
        vocab_size = 100
        logits_base = torch.randn(2, 5, vocab_size)

        def grad_norm(tau):
            stgs = STGS(
                vocab_size=vocab_size, stgs_hard=True,
                init_temperature=tau, device=device
            )
            stgs.eval()
            logits = logits_base.to(device).detach().requires_grad_(True)
            torch.manual_seed(1)
            _, one_hot, _, _ = stgs(logits)
            _rand_loss(one_hot).backward()
            return logits.grad.norm().item()

        norm_1   = grad_norm(1.0)
        norm_100 = grad_norm(100.0)
        ratio = norm_1 / norm_100

        assert ratio > 20.0, (
            f"Expected τ=1 gradient to be >20× larger than τ=100; "
            f"ratio={ratio:.1f}  (norm_1={norm_1:.4f}, norm_100={norm_100:.6f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gumbel noise scale
# ─────────────────────────────────────────────────────────────────────────────

class TestGumbelNoiseScale:
    """gumbel_noise_scale=0.0 → deterministic softmax; =1.0 → standard Gumbel-Softmax."""

    @pytest.mark.parametrize("noise_scale", [0.0, 0.5, 1.0, 2.0])
    def test_gradient_exists_at_all_noise_scales(self, device, noise_scale):
        """Gradient must flow for any non-negative gumbel_noise_scale. [FLOW]"""
        stgs = STGS(vocab_size=30, stgs_hard=True, device=device)
        stgs.eval()
        logits = torch.randn(1, 4, 30, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits, gumbel_noise_scale=noise_scale)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient at gumbel_noise_scale={noise_scale}"
        )

    def test_zero_noise_scale_is_deterministic(self, device):
        """gumbel_noise_scale=0.0 → same gradient on every forward call. [FLOW]"""
        stgs = STGS(vocab_size=30, stgs_hard=True, init_temperature=1.0, device=device)
        stgs.eval()
        logits_base = torch.randn(1, 4, 30, device=device)

        grads = []
        for _ in range(3):
            l = logits_base.detach().requires_grad_(True)
            _, one_hot, _, _ = stgs(l, gumbel_noise_scale=0.0)
            _rand_loss(one_hot).backward()
            grads.append(l.grad.clone())

        assert torch.allclose(grads[0], grads[1], atol=1e-6), (
            "gumbel_noise_scale=0 should give a deterministic gradient"
        )
        assert torch.allclose(grads[1], grads[2], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Logits normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestLogitsNormalization:
    """[FLOW] / [EDGE]: logits_normalize modes and their gradient implications."""

    @pytest.mark.parametrize("normalize", ["none", "center", "zscore"])
    def test_gradient_exists_with_normalization(self, device, normalize):
        """Gradient must flow for all normalization modes with typical logits. [FLOW]"""
        stgs = STGS(
            vocab_size=30, stgs_hard=True, logits_normalize=normalize, device=device
        )
        stgs.eval()
        logits = torch.randn(2, 5, 30, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient with logits_normalize='{normalize}'"
        )

    def test_zscore_uniform_logits_amplifies_gradient(self, device):
        """zscore + uniform logits (std≈0): gradient is numerically amplified. [EDGE]

        When all logit values are identical, std=0.  The normalised value is
            x_normed = (x − mean) / (std + eps) = 0 / eps
        The gradient ∂x_normed/∂x involves 1/(std+eps) ≈ 1/eps ≈ 1e12,
        producing extremely large (or NaN) gradients.
        This is a known edge case: avoid zscore normalisation with nearly-constant logits.
        """
        eps = 1e-12
        stgs = STGS(
            vocab_size=20, stgs_hard=True,
            logits_normalize="zscore", eps=eps, device=device
        )
        stgs.eval()
        logits_uniform = torch.ones(1, 3, 20, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits_uniform)
        _rand_loss(one_hot).backward()

        assert logits_uniform.grad is not None
        grad_norm = logits_uniform.grad.norm().item()
        # Either very large (1/eps amplification) or NaN — both indicate the edge case
        is_large_or_nan = (grad_norm > 1e3) or (grad_norm != grad_norm)  # nan check
        assert is_large_or_nan, (
            f"Expected huge or NaN gradient with zscore + uniform logits (eps={eps}); "
            f"got norm={grad_norm:.3e}"
        )

    def test_center_gradient_comparable_to_none(self, device):
        """'center' normalization only shifts logits; gradient magnitude should be close to 'none'."""
        vocab_size = 50
        logits_base = torch.randn(2, 5, vocab_size)
        norms = {}

        for norm in ["none", "center"]:
            stgs = STGS(
                vocab_size=vocab_size, stgs_hard=True,
                logits_normalize=norm, device=device
            )
            stgs.eval()
            logits = logits_base.to(device).detach().requires_grad_(True)
            torch.manual_seed(1)
            _, one_hot, _, _ = stgs(logits)
            _rand_loss(one_hot).backward()
            norms[norm] = logits.grad.norm().item()

        ratio = norms["none"] / (norms["center"] + 1e-12)
        assert 0.1 < ratio < 10.0, (
            f"'center' normalization caused unexpected gradient change; "
            f"ratio(none/center)={ratio:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  input_dropout: zeroes logit dimensions BEFORE softmax → [PARTIAL] in grad
# ─────────────────────────────────────────────────────────────────────────────

class TestInputDropout:
    """input_dropout applies F.dropout to x BEFORE Gumbel noise + softmax.

    The dropout mask multiplies the gradient chain at the x position,
    so ~p fraction of vocab dimensions get zero gradient. [PARTIAL]
    """

    @pytest.mark.parametrize("input_dropout", [0.0, 0.3, 0.5, 0.9])
    def test_gradient_flows_with_input_dropout(self, device, input_dropout):
        """Gradient must still flow in train mode at any input_dropout < 1.0. [PARTIAL→FLOW]"""
        torch.manual_seed(0)
        vocab_size = 200
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            input_dropout=input_dropout, device=device
        )
        stgs.train()
        logits = torch.randn(2, 8, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient with input_dropout={input_dropout}"
        )

    def test_input_dropout_creates_zeros_in_logit_grad(self, device):
        """input_dropout=0.5 → ~50% of vocab-position gradients are exactly zero. [PARTIAL]

        Gradient path: x → F.dropout(x, mask) → gumbel_noise → softmax → ...
        At the dropout layer: d(loss)/d(x[v]) = d(loss)/d(x_dropped[v]) * mask[v]/(1-p).
        When mask[v]=0: gradient is zero regardless of downstream signal.
        """
        torch.manual_seed(0)
        vocab_size = 2000
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            input_dropout=0.5, device=device
        )
        stgs.train()
        logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        zero_frac = (logits.grad == 0).float().mean().item()
        assert 0.3 < zero_frac < 0.7, (
            f"Expected ~50% zero-gradient positions with input_dropout=0.5; "
            f"got {zero_frac:.2%}"
        )

    def test_input_dropout_inactive_in_eval_mode(self, device):
        """In eval mode, input_dropout is disabled → full gradient coverage. [FLOW]"""
        vocab_size = 100
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            input_dropout=0.9, device=device
        )
        stgs.eval()
        logits = torch.randn(1, 4, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        zero_frac = (logits.grad == 0).float().mean().item()
        assert zero_frac < 0.05, (
            f"Unexpected zero-gradient positions in eval mode; zero_frac={zero_frac:.2%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  output_dropout: applied AFTER softmax — does NOT create zeros in logit grad
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDropout:
    """output_dropout applies F.dropout to y_soft AFTER softmax.

    Critical asymmetry with input_dropout:
      input_dropout  → mask is applied to x → d(loss)/d(x[v]) = 0 when mask[v]=0 [PARTIAL]
      output_dropout → mask is applied to y_soft; the softmax Jacobian COUPLES
                       all vocab positions, so d(loss)/d(x[v]) ≠ 0 even for
                       dropped y_soft positions → [FLOW] (not PARTIAL)

    Derivation: with output_dropout mask M,
        d(loss)/d(x[v]) = y_soft[v]*(M[v] − <M, y_soft>) / (τ*(1−p))
    which is non-zero whenever y_soft[v] > 0 (always true for softmax).

    With stgs_hard=True, the forward value of output_one_hot is still the
    hard one-hot (y_soft − y_soft.detach() = 0 numerically), but the
    backward gradient flows through the post-dropout y_soft.
    """

    @pytest.mark.parametrize("output_dropout", [0.0, 0.3, 0.5, 0.9])
    def test_gradient_flows_with_output_dropout(self, device, output_dropout):
        """Gradient must flow in train mode at any output_dropout. [FLOW]"""
        torch.manual_seed(0)
        vocab_size = 100
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            output_dropout=output_dropout, device=device
        )
        stgs.train()
        logits = torch.randn(2, 8, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient with output_dropout={output_dropout}"
        )

    def test_output_dropout_does_not_create_zero_logit_gradients(self, device):
        """Unlike input_dropout, output_dropout does NOT zero specific gradient positions. [FLOW]

        The softmax coupling ensures every x[v] receives a non-zero gradient
        signal regardless of which y_soft entries were dropped.
        """
        torch.manual_seed(0)
        vocab_size = 200
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            output_dropout=0.5, device=device
        )
        stgs.train()
        logits = torch.randn(2, 8, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)
        _rand_loss(one_hot).backward()

        zero_frac = (logits.grad == 0).float().mean().item()
        # Output dropout does NOT create structured zeros in logits.grad
        assert zero_frac < 0.05, (
            f"output_dropout should not produce zero-gradient positions "
            f"(softmax coupling); got zero_frac={zero_frac:.2%}.  "
            f"If this fails, check whether input_dropout was accidentally set."
        )

    def test_contrast_input_vs_output_dropout_zero_fraction(self, device):
        """input_dropout creates ~50% zeros in logits.grad; output_dropout does not.

        This is the key asymmetry: same nominal rate (p=0.5), very different
        gradient coverage.
        """
        torch.manual_seed(0)
        vocab_size = 2000

        def zero_grad_frac(input_dp, output_dp):
            stgs = STGS(
                vocab_size=vocab_size, stgs_hard=True,
                input_dropout=input_dp, output_dropout=output_dp, device=device
            )
            stgs.train()
            logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)
            torch.manual_seed(1)
            _, one_hot, _, _ = stgs(logits)
            _rand_loss(one_hot).backward()
            return (logits.grad == 0).float().mean().item()

        frac_input  = zero_grad_frac(input_dp=0.5, output_dp=0.0)
        frac_output = zero_grad_frac(input_dp=0.0, output_dp=0.5)

        assert 0.3 < frac_input < 0.7, (
            f"input_dropout=0.5 should give ~50% zero-gradient positions; "
            f"got {frac_input:.2%}"
        )
        assert frac_output < 0.05, (
            f"output_dropout=0.5 should give <5% zero-gradient positions; "
            f"got {frac_output:.2%}"
        )
        assert frac_input > frac_output + 0.25, (
            f"input_dropout should produce far more zeros than output_dropout; "
            f"frac_input={frac_input:.2%}, frac_output={frac_output:.2%}"
        )

    def test_output_dropout_breaks_simplex_in_forward(self, device):
        """In train mode, y_soft after output_dropout does NOT sum to 1 (not a simplex). [PARTIAL]

        F.dropout rescales surviving entries by 1/(1-p) but does not renormalise,
        so row sums ≠ 1.  The hard one-hot output_one_hot is still exactly one-hot
        (because y_soft − y_soft.detach() = 0 in the forward pass).
        """
        torch.manual_seed(0)
        vocab_size = 100
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            output_dropout=0.5, device=device
        )
        stgs.train()
        logits = torch.randn(1, 4, vocab_size, device=device)

        _, _, _, y_soft = stgs(logits)
        row_sums = y_soft.sum(dim=-1)

        assert not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-4
        ), "Expected output_dropout to break simplex constraint, but rows still sum to 1"

    def test_output_dropout_inactive_in_eval_mode(self, device):
        """In eval mode, output_dropout is inactive → y_soft is a valid distribution."""
        vocab_size = 50
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            output_dropout=0.9, device=device
        )
        stgs.eval()
        logits = torch.randn(1, 3, vocab_size, device=device)

        _, _, _, y_soft = stgs(logits)
        assert torch.allclose(
            y_soft.sum(dim=-1), torch.ones(1, 3, device=device), atol=1e-4
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  embsim-l2 hard selection
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbsimGradientFlow:
    """stgs_hard_method='embsim-l2': hard token selected by nearest embedding.

    Hard selection is inside torch.no_grad() → token IDs are discrete.
    Gradient flows via the straight-through path (same as categorical).
    """

    def test_embsim_l2_gradient_exists(self, device):
        """Gradient must flow with embsim-l2 + gumbel_soft probs. [FLOW]"""
        vocab_size, embed_dim = 50, 32
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            stgs_hard_method="embsim-l2",
            stgs_hard_embsim_probs="gumbel_soft",
            device=device,
        )
        stgs.eval()
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        logits = torch.randn(2, 5, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits, embedding_weights=E)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_embsim_l2_no_embedding_falls_back_to_categorical(self, device):
        """Without embedding_weights, embsim-l2 falls back to categorical → gradient flows. [FLOW]"""
        vocab_size = 30
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            stgs_hard_method="embsim-l2",
            device=device,
        )
        stgs.eval()
        logits = torch.randn(1, 4, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits)   # no embedding_weights
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    @pytest.mark.parametrize("embsim_probs", ["gumbel_soft", "input_logits"])
    def test_embsim_gradient_for_both_prob_sources(self, device, embsim_probs):
        """Both 'gumbel_soft' and 'input_logits' produce gradients via straight-through. [FLOW]"""
        vocab_size, embed_dim = 50, 32
        stgs = STGS(
            vocab_size=vocab_size, stgs_hard=True,
            stgs_hard_method="embsim-l2",
            stgs_hard_embsim_probs=embsim_probs,
            device=device,
        )
        stgs.eval()
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        logits = torch.randn(2, 5, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits, embedding_weights=E)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0), (
            f"All-zero gradient with stgs_hard_embsim_probs='{embsim_probs}'"
        )

    def test_high_temperature_embsim_causes_degenerate_token_selection(self, device):
        """At τ=100, nearly-uniform y_soft → soft_emb ≈ mean embedding → same token always.

        When y_soft is flat, soft_emb = Σ y_soft[v]*E[v] ≈ mean(E).
        The l2-nearest token to mean(E) is always the same "centroid" token,
        regardless of the learnable logits.  Learning signal is degraded
        even though gradients technically flow (straight-through is active).

        We verify: with diverse logit peaks, τ=1 gives many unique selected tokens
        while τ=100 gives far fewer unique tokens (selection collapsed to centroid).
        """
        torch.manual_seed(42)
        vocab_size, embed_dim = 100, 64
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        batch, seq = 8, 10

        # Logits with a strong peak at a different token per sequence position
        logits = torch.randn(batch, seq, vocab_size, device=device) * 0.1
        for s in range(seq):
            logits[:, s, (s * 7) % vocab_size] += 50.0   # clear per-position peak

        def unique_tokens(tau):
            stgs = STGS(
                vocab_size=vocab_size, stgs_hard=True,
                stgs_hard_method="embsim-l2",
                stgs_hard_embsim_probs="gumbel_soft",
                init_temperature=tau,
                device=device,
            )
            stgs.eval()
            with torch.no_grad():
                ids, _, _, _ = stgs(logits, embedding_weights=E)
            return ids.unique().numel()

        unique_low  = unique_tokens(1.0)
        unique_high = unique_tokens(100.0)

        # At τ=1 logit peaks dominate → embsim selects from distinct vocabulary regions
        # At τ=100 nearly-uniform y_soft → centroid collapse → fewer distinct tokens
        assert unique_low > unique_high, (
            f"Expected more diverse tokens at τ=1 than τ=100 (centroid collapse); "
            f"unique(τ=1)={unique_low}, unique(τ=100)={unique_high}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Run-script exact configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScriptConfig:
    """Exact run-script configuration from qwen3_600m_stgs_batch_k1-5-25-run.sh.

    Relevant CLI flags:
        --stgs_hard=True
        --stgs_hard_method=embsim-l2
        --stgs_hard_embsim_probs=gumbel_soft
        --logits_normalize=none
        --stgs_input_dropout=0.0
        --stgs_output_dropout=0.0
        --gumbel_noise_scale=1.0
        --temperature=100.0
        --learnable_temperature=True
    """

    def _make_stgs(self, vocab_size: int, device, tau: float = 100.0) -> STGS:
        return STGS(
            vocab_size=vocab_size,
            stgs_hard=True,
            stgs_hard_method="embsim-l2",
            stgs_hard_embsim_probs="gumbel_soft",
            logits_normalize="none",
            input_dropout=0.0,
            output_dropout=0.0,
            init_temperature=tau,
            learnable_temperature=True,
            device=device,
        )

    def test_gradient_flows_with_run_script_config(self, device):
        """Gradient must flow with the exact run-script STGS configuration. [FLOW / VANISH]"""
        torch.manual_seed(42)
        vocab_size, embed_dim = 200, 64
        stgs = self._make_stgs(vocab_size, device)
        stgs.eval()
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        logits = torch.randn(2, 10, vocab_size, device=device, requires_grad=True)

        _, one_hot, eff_tau, _ = stgs(logits, gumbel_noise_scale=1.0, embedding_weights=E)
        _rand_loss(one_hot).backward()

        assert logits.grad is not None, (
            "No gradient with run-script config (stgs_hard=True, τ=100, embsim-l2)"
        )
        assert not torch.all(logits.grad == 0), (
            "All-zero gradient with run-script config"
        )

    def test_run_script_gradient_is_significantly_attenuated(self, device):
        """At τ=100 the gradient norm is ≥10× smaller than at τ=1. [VANISH]

        The run-script learning_rate=1e-1 partially compensates, but this
        attenuation is an inherent consequence of the high initial temperature.
        """
        torch.manual_seed(42)
        vocab_size, embed_dim = 100, 64
        logits_base = torch.randn(2, 5, vocab_size)
        E = _fake_embeddings(vocab_size, embed_dim, device=device)

        def grad_norm(tau):
            stgs = self._make_stgs(vocab_size, device, tau=tau)
            stgs.eval()
            logits = logits_base.to(device).detach().requires_grad_(True)
            torch.manual_seed(1)
            _, one_hot, _, _ = stgs(logits, gumbel_noise_scale=1.0, embedding_weights=E)
            _rand_loss(one_hot).backward()
            return logits.grad.norm().item()

        norm_1   = grad_norm(1.0)
        norm_100 = grad_norm(100.0)
        ratio    = norm_1 / (norm_100 + 1e-12)

        assert ratio > 10.0, (
            f"Expected τ=1 gradient >10× larger than τ=100 with embsim-l2 config; "
            f"ratio={ratio:.1f}  (norm_1={norm_1:.4f}, norm_100={norm_100:.6f})"
        )

    def test_run_script_no_dropout_gives_full_gradient_coverage(self, device):
        """With input_dropout=0 and output_dropout=0: no zeros in logits.grad. [FLOW]"""
        torch.manual_seed(42)
        vocab_size, embed_dim = 50, 32
        stgs = self._make_stgs(vocab_size, device)
        stgs.eval()
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        logits = torch.randn(1, 5, vocab_size, device=device, requires_grad=True)

        _, one_hot, _, _ = stgs(logits, gumbel_noise_scale=1.0, embedding_weights=E)
        _rand_loss(one_hot).backward()

        zero_frac = (logits.grad == 0).float().mean().item()
        assert zero_frac < 0.05, (
            f"Unexpected zero-gradient fraction {zero_frac:.2%} with run-script config "
            f"(no input/output dropout expected)"
        )

    def test_run_script_embsim_token_selection_degeneracy_at_init_tau(self, device):
        """At τ=100 (run-script init), embsim-l2 may select similar tokens regardless of logits.

        This is the run-script regime where gradient VANISHING and TOKEN SELECTION
        DEGENERACY coincide.  Both effects are expected and managed by the learnable
        temperature annealing during optimisation, but callers should be aware.
        """
        torch.manual_seed(42)
        vocab_size, embed_dim = 100, 64
        E = _fake_embeddings(vocab_size, embed_dim, device=device)
        batch, seq = 4, 10

        # Strongly peaked logits at different positions per sequence step
        logits = torch.randn(batch, seq, vocab_size, device=device) * 0.1
        for s in range(seq):
            logits[:, s, (s * 7) % vocab_size] += 50.0

        def unique_tokens(tau):
            stgs = self._make_stgs(vocab_size, device, tau=tau)
            stgs.eval()
            with torch.no_grad():
                ids, _, _, _ = stgs(logits, gumbel_noise_scale=1.0, embedding_weights=E)
            return ids.unique().numel()

        unique_tau1   = unique_tokens(1.0)
        unique_tau100 = unique_tokens(100.0)

        assert unique_tau1 > unique_tau100, (
            f"Expected τ=1 to select more diverse tokens than τ=100; "
            f"unique(τ=1)={unique_tau1}, unique(τ=100)={unique_tau100}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Regime summary table (no assertions – serves as living documentation)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeSummaryDocs:
    """Non-asserting tests that document the full gradient-flow regime matrix.

    Run with -v to see the printed summary.
    """

    def test_print_regime_matrix(self, device, capsys):
        """Print a concise regime × outcome table for operator reference."""
        rows = [
            # (description,                       stgs_hard, dropout, input_dp, output_dp, tau, outcome)
            ("soft, no dropout",                   False,     0.0,     0.0,      0.0,      1.0, "[FLOW]   trivially differentiable"),
            ("hard, no dropout, τ=1",              True,      0.0,     0.0,      0.0,      1.0, "[FLOW]   straight-through"),
            ("hard, no dropout, τ=100",            True,      0.0,     0.0,      0.0,    100.0, "[VANISH] gradient ~100× attenuated"),
            ("hard, backprop dropout=1.0",         True,      1.0,     0.0,      0.0,      1.0, "[KILLED] gate=0, grad=zeros tensor"),
            ("soft, backprop dropout=1.0",         False,     1.0,     0.0,      0.0,      1.0, "[KILLED] no path, grad=None"),
            ("hard, input_dropout=0.5",            True,      0.0,     0.5,      0.0,      1.0, "[PARTIAL] ~50% zero grad positions"),
            ("hard, output_dropout=0.5",           True,      0.0,     0.0,      0.5,      1.0, "[FLOW]   softmax coupling; no zeros"),
            ("run-script (τ=100, embsim-l2)",      True,      0.0,     0.0,      0.0,    100.0, "[VANISH] + centroid selection"),
        ]
        header = f"{'Description':<45} {'Outcome'}"
        print()
        print("STGS Gradient-Flow Regime Matrix")
        print("=" * 80)
        print(header)
        print("-" * 80)
        for desc, *_, outcome in rows:
            print(f"{desc:<45} {outcome}")
        print("=" * 80)
        print()
        print("Note: [VANISH] does not mean broken – gradients exist and are")
        print("      correct for the Gumbel-Softmax Jacobian at high temperature.")
        print("      The run-script compensates with LR=1e-1 and learnable τ.")


class TestLossCoupledTempReg:
    """
    Tests for the loss-coupled temperature regularization:
        reg = λ * L_main.detach() / τ
        ∂reg/∂τ = -λ * L_main / τ²  < 0  →  gradient descent increases τ
    """

    def _make_tau(self, tau_val: float) -> torch.Tensor:
        """Return a scalar τ tensor that requires grad."""
        return torch.tensor(tau_val, dtype=torch.float32, requires_grad=True)

    # ------------------------------------------------------------------
    # 1. Gradient direction: ∂reg/∂τ < 0  →  optimizer increases τ
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("tau_val,loss_val,lam", [
        (1.0, 1.0, 0.1),
        (100.0, 2.0, 0.5),
        (0.5, 0.01, 1.0),
    ])
    def test_gradient_direction_pushes_tau_up(self, tau_val, loss_val, lam):
        """∂reg/∂τ must be strictly negative so gradient descent increases τ."""
        tau = self._make_tau(tau_val)
        loss_detached = torch.tensor(loss_val)  # already detached (no grad)
        coupling_reg = lam * loss_detached * (1.0 / tau.clamp(min=1e-6)).mean()
        coupling_reg.backward()
        # ∂(lam * L * mean(1/τ_i))/∂τ_i = -lam * L / (n * τ_i²)  < 0
        assert tau.grad is not None
        assert tau.grad.item() < 0.0, (
            f"Expected negative grad (push τ up), got {tau.grad.item():.6f} "
            f"at τ={tau_val}, L={loss_val}, λ={lam}"
        )

    # ------------------------------------------------------------------
    # 2. Analytic check (scalar τ, n=1): gradient equals -λ*L/τ²
    # ------------------------------------------------------------------
    def test_gradient_matches_analytic_formula(self):
        tau_val, loss_val, lam = 2.0, 3.0, 0.25
        tau = self._make_tau(tau_val)
        loss_detached = torch.tensor(loss_val)
        coupling_reg = lam * loss_detached * (1.0 / tau.clamp(min=1e-6)).mean()
        coupling_reg.backward()
        # For scalar τ (n=1): ∂/∂τ = -λ * L / τ²
        expected_grad = -lam * loss_val / (tau_val ** 2)
        assert abs(tau.grad.item() - expected_grad) < 1e-5, (
            f"Analytic grad={expected_grad:.6f}, got {tau.grad.item():.6f}"
        )

    # ------------------------------------------------------------------
    # 3. Reg scales linearly with loss magnitude
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("loss_scale", [0.5, 1.0, 2.0, 10.0])
    def test_reg_proportional_to_loss(self, loss_scale):
        """reg = λ * L * mean(1/τ) must scale linearly with L."""
        lam, tau_val = 0.1, 5.0
        base_loss = torch.tensor(1.0)
        scaled_loss = torch.tensor(loss_scale)
        tau = torch.tensor(tau_val)
        reg_base = lam * base_loss * (1.0 / tau.clamp(min=1e-6)).mean()
        reg_scaled = lam * scaled_loss * (1.0 / tau.clamp(min=1e-6)).mean()
        ratio = reg_scaled.item() / reg_base.item()
        assert abs(ratio - loss_scale) < 1e-5, (
            f"Expected ratio={loss_scale}, got {ratio:.6f}"
        )

    # ------------------------------------------------------------------
    # 4. Reg is zero (disabled) when λ=0
    # ------------------------------------------------------------------
    def test_disabled_at_zero_lambda(self):
        tau = self._make_tau(5.0)
        loss_detached = torch.tensor(2.5)
        lam = 0.0
        coupling_reg = lam * loss_detached * (1.0 / tau.clamp(min=1e-6)).mean()
        assert coupling_reg.item() == 0.0

    # ------------------------------------------------------------------
    # 5. detach() check: gradient does NOT flow back through loss term
    # ------------------------------------------------------------------
    def test_loss_is_detached_no_grad_through_logits(self):
        """
        When loss is computed from logits and then detached before multiplying by mean(1/τ),
        the logits must receive zero gradient from the coupling reg (only τ gets a grad).
        Use a squared loss so it is always non-negative (matching real hinge/CE usage).
        """
        logits = torch.randn(1, 5, 10, requires_grad=True)
        # Simulate a non-negative loss computed from logits (like hinge or CE)
        loss = logits.pow(2).mean()    # always > 0, has grad_fn through logits
        loss_d = loss.detach()         # detached — no path back to logits
        tau = self._make_tau(2.0)
        lam = 0.1
        coupling_reg = lam * loss_d * (1.0 / tau.clamp(min=1e-6)).mean()
        coupling_reg.backward()
        # τ should have a negative gradient (push up); logits should NOT have any gradient
        assert tau.grad is not None and tau.grad.item() < 0.0
        assert logits.grad is None, "coupling_reg must not back-propagate through the loss path"

    # ------------------------------------------------------------------
    # 6. Soft floor: smaller τ → larger magnitude gradient (stronger push)
    # ------------------------------------------------------------------
    def test_soft_floor_stronger_push_at_low_tau(self):
        """|∂reg/∂τ_i| = λ*L/(n*τ_i²) grows as τ shrinks — natural soft floor."""
        loss_val, lam = 1.0, 0.1
        loss_d = torch.tensor(loss_val)
        tau_high = self._make_tau(10.0)
        tau_low = self._make_tau(1.0)
        (lam * loss_d * (1.0 / tau_high.clamp(min=1e-6)).mean()).backward()
        (lam * loss_d * (1.0 / tau_low.clamp(min=1e-6)).mean()).backward()
        grad_high = abs(tau_high.grad.item())
        grad_low = abs(tau_low.grad.item())
        assert grad_low > grad_high, (
            f"Expected |∂reg/∂τ| larger at τ=1 than τ=10, got {grad_low:.4f} vs {grad_high:.4f}"
        )

    # ------------------------------------------------------------------
    # 7. Decoupled positions: each τ_i gets independent gradient; no compensation
    # ------------------------------------------------------------------
    def test_decoupled_positions_independent_gradients(self):
        """
        With n per-position temperatures, a collapsed τ_0≈0 gets a much larger
        gradient magnitude than a healthy τ_1=100 — the reg cannot be masked.
        Analytic: ∂reg/∂τ_i = -λ * L / (n * τ_i²)
        """
        lam, loss_val, n = 0.1, 2.0, 2
        loss_d = torch.tensor(loss_val)
        tau_collapsed = torch.tensor(0.1, requires_grad=True)   # near-zero
        tau_healthy = torch.tensor(100.0, requires_grad=True)   # high, healthy

        # Simulate the decoupled case: _eff_temp = [tau_collapsed, tau_healthy]
        eff_temp = torch.stack([tau_collapsed, tau_healthy])
        coupling_reg = lam * loss_d * (1.0 / eff_temp.clamp(min=1e-6)).mean()
        coupling_reg.backward()

        grad_collapsed = abs(tau_collapsed.grad.item())
        grad_healthy = abs(tau_healthy.grad.item())

        # ∂reg/∂τ_collapsed = -λ*L/(n*τ_c²)  >> ∂reg/∂τ_healthy
        expected_collapsed = lam * loss_val / (n * 0.1 ** 2)
        expected_healthy = lam * loss_val / (n * 100.0 ** 2)
        assert abs(grad_collapsed - expected_collapsed) < 1e-4
        assert abs(grad_healthy - expected_healthy) < 1e-8
        # Collapsed position receives >> 1000× stronger push than healthy
        assert grad_collapsed > 1000 * grad_healthy, (
            f"Collapsed pos grad={grad_collapsed:.4f} should dwarf healthy={grad_healthy:.8f}"
        )
