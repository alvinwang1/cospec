"""CPU checks of the GPU scripts' logic with tiny random Qwen3 models."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu"))

from collect_traces import position_drafts  # noqa: E402
from common import CachedLM, verify_forward, zero_cache  # noqa: E402
from sd_decode import speculative_generate  # noqa: E402


def tiny_qwen(seed, layers=2):
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=512)
    return transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def models():
    target = tiny_qwen(0)
    # A drafter that agrees with the target often: same weights, plus noise.
    drafter = tiny_qwen(0)
    with torch.no_grad():
        for p in drafter.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return target, drafter


@pytest.mark.parametrize("k", [1, 3, 6])
def test_speculative_generate_is_lossless(models, k):
    target, drafter = models
    prompt = list(range(5, 17))
    ref, _ = speculative_generate(target, drafter, prompt, 60, 0, eos=set())
    out, st = speculative_generate(target, drafter, prompt, 60, k, eos=set())
    assert out == ref
    assert st["accepted"] > 0  # the near-copy drafter should get some right


def test_cached_lm_matches_full_forward(models):
    target, _ = models
    seq = list(range(3, 30))
    lm = CachedLM(target)
    lm.extend(seq[:10])
    lm.extend(seq[10:20])
    lm.crop(15)
    step = lm.extend(seq[15:])
    with torch.no_grad():
        full = target(torch.tensor([seq])).logits[0].float()
    assert torch.allclose(step, full[15:], atol=1e-4)


def test_position_drafts_agree_with_sd_steps(models):
    """At positions SD visits, the trace's acceptance equals what SD accepted."""
    target, drafter = models
    prompt = list(range(7, 20))
    K = 4
    y, st = speculative_generate(target, drafter, prompt, 50, K, eos=set())
    conf, acc = position_drafts(drafter, prompt, y, K)
    assert conf.shape == (len(y), K) and ((conf > 0) & (conf <= 1)).all()
    p = 1
    for a, c in zip(st["acc"], st["confs"]):
        assert acc[p] == a
        assert np.allclose(conf[p], c, atol=1e-4)
        p += a + 1


def test_zero_cache_verify_and_crop(models):
    target, _ = models
    cache = zero_cache(target, batch=3, ctx=20)
    verify_forward(target, cache, 3, 20, q=5)
    assert cache.get_seq_length() == 20
