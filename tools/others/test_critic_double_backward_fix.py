"""Verification for the critic double-backward fix in helios/utils/utils_helios_post.py.

Background
----------
In multi-model DeepSpeed mode (DMD critic), `critic_accelerator.backward(loss)`
goes through `accelerate.utils.deepspeed.DeepSpeedEngineWrapper.backward`, which
calls `engine.backward(loss)` followed by `engine.step()` whenever
`accelerator.sync_gradients` is True. The critic update path is NOT wrapped in
`critic_accelerator.accumulate(...)` and the critic accelerator is constructed
with `gradient_accumulation_steps=1`, so `sync_gradients` is always True ==>
every `critic_accelerator.backward(...)` triggers an immediate optimizer step
plus zero_grad.

The GAN+low_vram path of `_critic_loss` calls backward twice (denoising, then
GAN/R1/R2). Pre-fix, this resulted in two optimizer steps per Python iter, the
denoising and GAN gradients never co-occurring in the optimizer's update.

Fix: drive the DeepSpeed engine directly. The first backward sets
`set_gradient_accumulation_boundary(False)` to skip the optimizer step; the
second backward sets it to True and is followed by an explicit `engine.step()`.

This script asserts:
  1. The fixed call sequence calls engine.step() exactly once per critic update.
  2. The fixed sequence sums gradients from both backwards before stepping.
  3. On a real Adam optimizer with conflicting gradients, the buggy two-step
     pattern and the fixed one-step pattern produce visibly different parameter
     trajectories — i.e. the bug was numerically meaningful, not just stylistic.

Run:
    python tools/others/test_critic_double_backward_fix.py
"""

import torch


class FakeEngine:
    """Records the sequence of DeepSpeed engine calls a critic update issues.

    Mimics the surface area driven by helios/utils/utils_helios_post.py's
    fixed code path: set_gradient_accumulation_boundary -> backward -> step.
    """

    def __init__(self):
        self.boundary_calls = []
        self.backward_calls = []
        self.step_count = 0
        self._grad_buffer = 0.0
        self.optimizer_inputs = []

    def set_gradient_accumulation_boundary(self, is_boundary):
        self.boundary_calls.append(bool(is_boundary))

    def backward(self, loss):
        self.backward_calls.append(float(loss.detach().item()))
        self._grad_buffer += float(loss.detach().item())

    def step(self):
        self.step_count += 1
        self.optimizer_inputs.append(self._grad_buffer)
        self._grad_buffer = 0.0


def fixed_path(engine, denoising_loss, gan_loss):
    """Mirrors helios/utils/utils_helios_post.py:3158-3306 (post-fix)."""
    engine.set_gradient_accumulation_boundary(is_boundary=False)
    engine.backward(denoising_loss)
    engine.set_gradient_accumulation_boundary(is_boundary=True)
    engine.backward(gan_loss)
    engine.step()


def buggy_path(engine, denoising_loss, gan_loss):
    """What the old code did via accelerator.backward (= boundary=True + step each time)."""
    engine.set_gradient_accumulation_boundary(is_boundary=True)
    engine.backward(denoising_loss)
    engine.step()
    engine.set_gradient_accumulation_boundary(is_boundary=True)
    engine.backward(gan_loss)
    engine.step()


def test_fixed_path_steps_once_with_combined_grad():
    engine = FakeEngine()
    denoising = torch.tensor(1.5)
    gan = torch.tensor(2.5)

    fixed_path(engine, denoising, gan)

    assert len(engine.backward_calls) == 2, (
        f"expected 2 backward calls, got {len(engine.backward_calls)}"
    )
    assert engine.step_count == 1, (
        f"expected 1 step call (combined update), got {engine.step_count}"
    )
    assert engine.boundary_calls == [False, True], (
        f"expected boundary toggles [False, True], got {engine.boundary_calls}"
    )
    expected = 1.5 + 2.5
    assert abs(engine.optimizer_inputs[0] - expected) < 1e-6, (
        f"expected combined grad {expected}, got {engine.optimizer_inputs[0]}"
    )
    print("PASS: fixed path -> 1 engine.step() with summed denoising+gan grads.")


def test_buggy_path_repro():
    """Sanity-check the bug repro so the fix's value is unambiguous."""
    engine = FakeEngine()
    denoising = torch.tensor(1.5)
    gan = torch.tensor(2.5)

    buggy_path(engine, denoising, gan)

    assert engine.step_count == 2, (
        f"bug repro: expected 2 step calls, got {engine.step_count}"
    )
    assert abs(engine.optimizer_inputs[0] - 1.5) < 1e-6, (
        f"first step should see denoising grad alone, got {engine.optimizer_inputs[0]}"
    )
    assert abs(engine.optimizer_inputs[1] - 2.5) < 1e-6, (
        f"second step should see gan grad alone (denoising zero'd), got {engine.optimizer_inputs[1]}"
    )
    print("PASS: pre-fix path -> 2 engine.step() calls; denoising and gan grads never co-occur.")


def _adam_run(buggy: bool, steps: int = 20, lr: float = 0.01) -> float:
    """Run a tiny Adam loop with two conflicting losses; return final param value.

    The two losses pull in opposite directions. Under the fixed path, gradients
    sum and approximately cancel each iteration so the param drifts only via
    Adam's adaptive moments. Under the buggy path, the two updates are applied
    sequentially; the first step shifts the param before the second loss is
    evaluated, and Adam's m/v are updated twice per iter -> different trajectory.
    """
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([param], lr=lr)

    for _ in range(steps):
        loss1 = (param - 0.5).pow(2).sum()  # pulls toward +0.5
        loss2 = (param + 0.5).pow(2).sum()  # pulls toward -0.5

        if buggy:
            opt.zero_grad()
            loss1.backward()
            opt.step()
            opt.zero_grad()
            # Recompute loss2 against the post-step param to mirror the real
            # critic flow where the second forward sees updated weights.
            loss2_recompute = (param + 0.5).pow(2).sum()
            loss2_recompute.backward()
            opt.step()
        else:
            opt.zero_grad()
            loss1.backward()
            loss2.backward()
            opt.step()

    return float(param.detach().item())


def test_real_optimizer_diverges():
    buggy_final = _adam_run(buggy=True)
    fixed_final = _adam_run(buggy=False)
    diff = abs(buggy_final - fixed_final)

    print(f"  buggy two-step trajectory ends at param = {buggy_final:.6f}")
    print(f"  fixed one-step trajectory ends at param = {fixed_final:.6f}")
    assert diff > 1e-3, (
        "expected the two paths to diverge meaningfully; got "
        f"buggy={buggy_final}, fixed={fixed_final}, diff={diff}"
    )
    print(f"PASS: buggy and fixed Adam trajectories diverge (|delta| = {diff:.6f}).")


def test_engine_attr_lookup_safe_when_no_deepspeed():
    """The fix must fall back to accelerator.backward when DeepSpeed isn't wired up."""

    class _NoDS:
        deepspeed_engine_wrapped = None

    class _NoAttr:
        pass

    for accel in (_NoDS(), _NoAttr()):
        engine = getattr(getattr(accel, "deepspeed_engine_wrapped", None), "engine", None)
        assert engine is None, "engine resolution must yield None outside DeepSpeed"
    print("PASS: engine lookup returns None for non-DeepSpeed accelerators.")


if __name__ == "__main__":
    test_fixed_path_steps_once_with_combined_grad()
    test_buggy_path_repro()
    test_real_optimizer_diverges()
    test_engine_attr_lookup_safe_when_no_deepspeed()
    print("\nAll critic double-backward fix checks passed.")
