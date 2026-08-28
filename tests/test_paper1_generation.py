from types import SimpleNamespace

from ccpu.paper1.generation import _eos_token_ids, select_device


def _torch(*, cuda: bool, xpu: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
    )


def test_auto_device_prefers_cuda_then_xpu_then_cpu():
    assert select_device(_torch(cuda=True, xpu=True), "auto") == "cuda"
    assert select_device(_torch(cuda=False, xpu=True), "auto") == "xpu"
    assert select_device(_torch(cuda=False, xpu=False), "auto") == "cpu"


def test_explicit_device_is_preserved():
    assert select_device(_torch(cuda=False, xpu=False), "xpu") == "xpu"


def test_eos_tokens_include_model_specific_end_of_turn():
    assert _eos_token_ids(1, [1, 106]) == frozenset({1, 106})
    assert _eos_token_ids(None, 106) == frozenset({106})
