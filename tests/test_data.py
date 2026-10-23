"""The prompt lists under data/ make the repo self-contained: no dataset
download (and no ``datasets`` dependency) at run time."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # abliterate.py imports torch at module level

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import abliterate  # noqa: E402


def test_data_covers_the_default_prompt_counts():
    defaults = abliterate.parse_args([])
    harmful_train, harmless_train, harmful_test = abliterate.load_prompts(
        defaults.n_train, defaults.n_test
    )
    assert len(harmful_train) == defaults.n_train
    assert len(harmless_train) == defaults.n_train
    assert len(harmful_test) == defaults.n_test
    every = harmful_train + harmless_train + harmful_test
    assert all(isinstance(t, str) and t.strip() for t in every)


def test_held_out_test_prompts_are_disjoint_from_train():
    harmful_train, _, harmful_test = abliterate.load_prompts(10_000, 10_000)
    assert not set(harmful_train) & set(harmful_test)


def test_probe_prompts_are_fresh():
    """Probes (used to pick layer/strength) must come from neither the prompts
    that built the direction nor the held-out test set."""
    n_train, n_probes = 256, 24
    harmful_train, _, harmful_test = abliterate.load_prompts(n_train, 32)
    probes = abliterate.probe_prompts(n_train, n_probes)
    assert len(probes) == n_probes
    assert not set(probes) & set(harmful_train)
    assert not set(probes) & set(harmful_test)


@pytest.mark.parametrize("extra", [0, 1, 1000])
def test_exhausted_probe_pool_never_falls_back_to_held_out_test(extra):
    pool = abliterate._read_prompts(abliterate.DATA_DIR / "harmful_behaviors_train.csv")

    with pytest.raises(ValueError, match="--n-train"):
        abliterate.probe_prompts(len(pool) + extra, 24)


def test_small_remaining_probe_pool_stays_disjoint():
    pool = abliterate._read_prompts(abliterate.DATA_DIR / "harmful_behaviors_train.csv")
    probes = abliterate.probe_prompts(len(pool) - 1, 24)

    assert probes == pool[-1:]
    assert not set(probes) & set(pool[:-1])
    assert not set(probes) & set(
        abliterate._read_prompts(abliterate.DATA_DIR / "harmful_behaviors_test.csv")
    )


def test_exhausted_probe_pool_is_rejected_before_loading_a_model(monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def forbidden_load(*args, **kwargs):
        pytest.fail("invalid probe configuration must not load a model or tokenizer")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden_load)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden_load)
    pool = abliterate._read_prompts(abliterate.DATA_DIR / "harmful_behaviors_train.csv")
    args = abliterate.parse_args(["--n-train", str(len(pool)), "--device", "cpu"])

    with pytest.raises(ValueError, match="--n-train"):
        abliterate.run(args)
