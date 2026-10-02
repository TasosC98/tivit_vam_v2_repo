"""A non-finite training batch must not break BatchNorm's running statistics.

Needs torch (runs on the server; skipped where torch is not installed):
    python -m pytest tests/test_train_guard.py
"""
import pytest

torch = pytest.importorskip("torch")

from pianovam_vision.train import BatchNormGuard  # noqa: E402


def small_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                               torch.nn.Conv2d(4, 4, 3), torch.nn.BatchNorm2d(4))


def test_restore_undoes_a_bad_batch():
    model = small_model().train()
    model(torch.randn(2, 3, 9, 9))                  # normal statistics first
    guard = BatchNormGuard(model)
    assert len(guard.buffers) == 6 and guard.finite()
    good = [b.clone() for b in guard.buffers]

    guard.snapshot()
    x = torch.randn(2, 3, 9, 9)
    x[0, 0, 4, 4] = float("inf")                    # e.g. an fp16 overflow
    model(x)
    assert not guard.finite()                       # what broke E6's validation
    guard.restore()
    assert guard.finite()
    assert all(torch.equal(a, b) for a, b in zip(good, guard.buffers))


def test_eval_outputs_stay_finite_after_a_restored_bad_batch():
    model = small_model().train()
    guard = BatchNormGuard(model)
    for i in range(5):
        guard.snapshot()
        x = torch.randn(2, 3, 9, 9)
        if i == 2:
            x[1, 2, 0, 0] = float("nan")
        model(x)
        if not guard.finite():
            guard.restore()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(1, 3, 9, 9))).all()
