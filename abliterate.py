#!/usr/bin/env python3
"""Abliterate a small instruct model in a few minutes.

Default target is Qwen/Qwen2.5-1.5B-Instruct, which runs on an Apple-silicon Mac
(MPS), a modest NVIDIA GPU, or CPU (slow). Pure ``transformers``; no TransformerLens.

Notes:
  * Sanitised by default. The script reports refusal *rates* on a held-out set
    and shows one benign prompt before/after so you can see the model still
    works. Harmful completions are never printed unless you pass
    ``--show-outputs`` (and even then they are truncated).
  * Deterministic (greedy decoding), so runs are comparable.
  * The layer and the ablation strength α are chosen by trying real edits:
    candidates are applied permanently, verified on fresh probe prompts
    (never the eval set), undone, and the best one is re-applied — fewest
    refusals, then most real answers, then the weakest edit. Pin either with
    ``--layer`` / ``--alpha``.

Usage:
    python abliterate.py                           # full run, ~6 min on Apple silicon; saves the edited
                                                   # model to qwen-abliterated/ for chat.py
    python abliterate.py --n-train 64 --n-test 16 --max-new-tokens 48 --max-candidates 3 --top-k 1   # ~2 min
    python abliterate.py --no-save                 # don't write the edited model (~6 GB)

Prompt sets (mlabonne/harmful_behaviors and mlabonne/harmless_alpaca, the same
ones used in Labonne's post) are committed under data/, so this runs with no
dataset download; fetch_datasets.py regenerates them from the Hub.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.simplefilter("ignore")                       # urllib3/LibreSSL, dtype deprecation, etc.
sys.stdout.reconfigure(line_buffering=True)           # progress shows up live even through a pipe/tee
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abliteration_core import (  # noqa: E402
    LayerScore,
    ablate_activation,
    candidate_layers,
    compute_refusal_directions,
    is_refusal,
    on_topic_rate,
    orthogonalize_weight,
    refusal_rate,
)

BENIGN_SANITY_PROMPT = "Explain in two sentences why the sky is blue."
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "qwen-abliterated"   # what chat.py loads by default
DATA_DIR = Path(__file__).resolve().parent / "data"                       # committed with the repo

# -- console styling -----------------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s): return _c("1", s)
def teal(s): return _c("36", s)
def green(s): return _c("32", s)
def red(s): return _c("31", s)
def dim(s): return _c("2", s)


def banner(msg: str, sub: str = "") -> None:
    line = "─" * 70
    print(f"\n{teal(line)}\n{bold(msg)}" + (f"\n{dim(sub)}" if sub else "") + f"\n{teal(line)}")


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    # fp16 overflows on Qwen2.5; bf16 is slower than fp32 on Apple silicon for small models.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _read_prompts(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row["text"] for row in csv.DictReader(f)]


def load_prompts(n_train: int, n_test: int) -> tuple[list[str], list[str], list[str]]:
    """Read the prompt lists from data/; nothing is downloaded at run time."""
    harmful_train = _read_prompts(DATA_DIR / "harmful_behaviors_train.csv")[:n_train]
    harmless_train = _read_prompts(DATA_DIR / "harmless_alpaca.csv")[:n_train]
    harmful_test = _read_prompts(DATA_DIR / "harmful_behaviors_test.csv")[:n_test]
    return harmful_train, harmless_train, harmful_test


def probe_prompts(n_train: int, n_probes: int) -> list[str]:
    """Fresh harmful prompts (past the ones used for the direction) for selecting
    the layer and strength, so the test split stays fully held out."""
    pool = _read_prompts(DATA_DIR / "harmful_behaviors_train.csv")
    tail = pool[n_train : n_train + n_probes]
    if not tail:
        raise ValueError(
            f"no unused training prompts remain for layer/strength probes; "
            f"reduce --n-train below {len(pool)} to keep the test set held out"
        )
    return tail


def chat_encode(tokenizer, prompts: list[str], device: torch.device):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def last_token_activations(model, tokenizer, prompts: list[str], device, batch_size: int):
    """[n_prompts, n_layers, d_model] residual-stream activations at the final token."""
    chunks = []
    for i in range(0, len(prompts), batch_size):
        enc = chat_encode(tokenizer, prompts[i : i + batch_size], device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states[0] is the embedding output; [1..L] are post-block residuals.
        hs = torch.stack(out.hidden_states[1:], dim=1)  # [b, L, T, d]
        chunks.append(hs[:, :, -1, :].float().cpu())    # left-padded => last pos is real
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], device, max_new_tokens: int, batch_size: int):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = chat_encode(tokenizer, prompts[i : i + batch_size], device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :]
        outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return outs


def decoder_layers(model):
    # Works for Llama/Qwen/Mistral/Gemma style HF models.
    return model.model.layers


@contextmanager
def inference_time_ablation(model, direction: torch.Tensor, alpha: float = 1.0):
    """Temporarily subtract ``alpha`` × the refusal direction from every block's output."""
    handles = []

    def hook(_module, _inputs, output, _alpha=alpha):
        if isinstance(output, tuple):
            return (ablate_activation(output[0], direction, _alpha),) + tuple(output[1:])
        return ablate_activation(output, direction, _alpha)

    for layer in decoder_layers(model):
        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def orthogonalize_model(model, direction: torch.Tensor, alpha: float = 1.0) -> None:
    """Permanently remove ``alpha`` × the ability to write along ``direction``."""
    # Embedding rows are vectors in residual space -> axis=1.
    emb = model.model.embed_tokens.weight
    emb.data = orthogonalize_weight(emb.data, direction, axis=1, alpha=alpha)
    # nn.Linear weights are [out, in]; the *output* (rows) is written into the residual
    # stream -> axis=0. Explicit because o_proj is square and cannot be inferred.
    for layer in decoder_layers(model):
        o = layer.self_attn.o_proj.weight
        o.data = orthogonalize_weight(o.data, direction, axis=0, alpha=alpha)
        d = layer.mlp.down_proj.weight
        d.data = orthogonalize_weight(d.data, direction, axis=0, alpha=alpha)


def writer_weights(model):
    """Every weight matrix that gets edited, as (name, tensor) pairs."""
    yield "embed_tokens", model.model.embed_tokens.weight
    for i, layer in enumerate(decoder_layers(model)):
        yield f"layers.{i}.o_proj", layer.self_attn.o_proj.weight
        yield f"layers.{i}.down_proj", layer.mlp.down_proj.weight


def snapshot_writer_weights(model) -> dict:
    return {name: w.detach().to("cpu", copy=True) for name, w in writer_weights(model)}


def restore_writer_weights(model, snapshot: dict) -> None:
    for name, w in writer_weights(model):
        w.data = snapshot[name].to(device=w.device, dtype=w.dtype)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"],
                    help="auto = bf16 on CUDA, fp32 elsewhere (fp16 overflows on Qwen2.5)")
    ap.add_argument("--n-train", type=int, default=256, help="prompts per class for the direction")
    ap.add_argument("--n-test", type=int, default=64, help="held-out harmful prompts for the before/after report")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=6, help="layers to preview with hooks")
    ap.add_argument("--top-k", type=int, default=3, help="previewed layers to try with verified real edits")
    ap.add_argument("--layer", type=int, default=None, help="skip the search and use this layer")
    ap.add_argument("--alpha", type=float, default=None,
                    help="projection strength to remove (1.0 = full ablation). Default: sweep and keep the "
                         "weakest strength that still removes refusals")
    ap.add_argument("--save", type=Path, default=DEFAULT_SAVE_DIR,
                    help="where to write the edited model so chat.py can load it (default: qwen-abliterated next to this script)")
    ap.add_argument("--no-save", action="store_true", help="don't write the edited model to disk")
    ap.add_argument("--show-outputs", action="store_true",
                    help="print (truncated) held-out completions. Off by default: these can be harmful.")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print(red("\nstopped by user"))
        return 130
    return 0


def run(args) -> None:
    # Validate the selection pool before downloading/loading the model. Never
    # fall back to the held-out test split when the training pool is exhausted.
    probe = probe_prompts(args.n_train, min(24, args.n_test))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    t_start = time.time()

    # ---- 1. model ----------------------------------------------------------
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    attn = "eager" if device.type == "mps" else "sdpa"   # MPS SDPA -> NaN on left-padded rows
    banner("1 · Load the model", f"{args.model} on {device.type} ({str(dtype).replace('torch.', '')}, {attn} attention)")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attn
    ).to(device).eval()
    n_layers = len(decoder_layers(model))
    print(f"{n_layers} layers, hidden size {model.config.hidden_size}  {dim(f'({time.time() - t0:.1f}s)')}")

    # ---- 2. prompts --------------------------------------------------------
    banner("2 · Load the example questions", "from data/ in this repo — no download")
    harmful_train, harmless_train, harmful_test = load_prompts(args.n_train, args.n_test)
    print(f"{len(harmful_train)} harmful + {len(harmless_train)} harmless to find the direction; "
          f"{len(harmful_test)} different harmful ones held back for scoring")

    # ---- 3. before ---------------------------------------------------------
    banner("3 · Before: how often does the original model refuse?")
    t0 = time.time()
    base_outputs = generate(model, tokenizer, harmful_test, device, args.max_new_tokens, args.batch_size)
    base_rate = refusal_rate(base_outputs)
    base_answered = on_topic_rate(harmful_test, base_outputs)
    base_benign = generate(model, tokenizer, [BENIGN_SANITY_PROMPT], device, args.max_new_tokens, 1)[0]
    print(f"baseline refusal rate: {red(bold(f'{base_rate:.0%}'))}  {dim(f'({time.time() - t0:.1f}s)')}")
    print(f"real on-topic answers: {base_answered:.0%}  {dim('(not a refusal, 8+ words, shares content words with the question)')}")
    print(f"{dim('harmless check')} \"{BENIGN_SANITY_PROMPT}\"\n  → {base_benign.strip()[:200]}")

    # ---- 4. direction ------------------------------------------------------
    banner("4 · Find the \"refuse\" direction", "last-token residual activation per question; harmful minus harmless")
    t0 = time.time()
    harmful_acts = last_token_activations(model, tokenizer, harmful_train, device, args.batch_size)
    harmless_acts = last_token_activations(model, tokenizer, harmless_train, device, args.batch_size)
    directions = compute_refusal_directions(harmful_acts, harmless_acts)
    print(f"one candidate direction per layer: {tuple(directions.shape)}  {dim(f'({time.time() - t0:.1f}s)')}")

    # ---- 5. shortlist -------------------------------------------------------
    banner("5 · Shortlist the layers", "cheap hook preview ranks the candidates; real edits decide in the next step")
    if args.layer is not None:
        shortlist = [args.layer]
        print(f"using user-specified layer {args.layer}")
    else:
        cands = candidate_layers(n_layers)
        raw_norms = (harmful_acts.mean(0) - harmless_acts.mean(0)).norm(dim=-1)
        cands = sorted(cands, key=lambda l: -float(raw_norms[l]))[: args.max_candidates]
        scores: list[LayerScore] = []
        for layer in sorted(cands):
            d = directions[layer].to(device=device, dtype=dtype)
            with inference_time_ablation(model, d):
                outs = generate(model, tokenizer, probe, device, min(32, args.max_new_tokens), args.batch_size)
            s = LayerScore(
                layer=layer,
                refusal_rate=refusal_rate(outs),
                on_topic=on_topic_rate(probe, outs),
            )
            scores.append(s)
            bar = "█" * int(round(s.refusal_rate * 20)) + "░" * (20 - int(round(s.refusal_rate * 20)))
            print(f"  layer {layer:>2}: refusal {s.refusal_rate:>4.0%}  on-topic {s.on_topic:>4.0%}  "
                  f"(depth {layer / n_layers:.0%})  {red(bar)}")
        ranked = sorted(scores, key=lambda s: (s.refusal_rate, -s.on_topic, s.layer))
        shortlist = [s.layer for s in ranked[: max(1, args.top_k)]]
        print(f"-> shortlist for real edits: {shortlist}  {dim('(preview ranking only — hooks and real edits disagree)')}")

    # ---- 6. edit trials -----------------------------------------------------
    banner("6 · Try real edits", "each candidate is a permanent edit verified on fresh probes and undone; the best stays")
    snapshot = snapshot_writer_weights(model)
    t0 = time.time()
    trials: list[tuple[int, float, float, float]] = []   # (layer, alpha, refusal, on_topic)

    def try_edit(layer: int, a: float) -> float:
        d = directions[layer].to(device=device, dtype=dtype)
        restore_writer_weights(model, snapshot)
        orthogonalize_model(model, d, a)
        outs = generate(model, tokenizer, probe, device, args.max_new_tokens, args.batch_size)
        ref, on = refusal_rate(outs), on_topic_rate(probe, outs)
        print(f"  layer {layer:>2} × α = {a:<4}: refusal {ref:>4.0%}  on-topic {on:>4.0%}")
        trials.append((layer, a, ref, on))
        return ref

    for layer in shortlist:
        # Gate with full strength: if α = 1 can't remove every probe refusal,
        # no weaker edit can either, so the layer is out after one trial.
        if try_edit(layer, 1.0 if args.alpha is None else args.alpha) > 0.0:
            continue
        if args.alpha is None:
            for a in (0.8, 0.6):
                if try_edit(layer, a) > 0.0:
                    break   # too weak; the previous strength was this layer's best
    if not any(t[2] == 0.0 for t in trials):
        # nothing fully removed refusals; give the most promising layer one stronger try
        fallback = min(trials, key=lambda t: (t[2], -t[3], t[0]))
        print(dim("no edit removed every probe refusal; trying one stronger edit"))
        try_edit(fallback[0], 1.25)
    # fewest refusals, then most real answers, then the weakest edit, then the shallowest layer
    chosen_layer, alpha, _, _ = min(trials, key=lambda t: (t[2], -t[3], t[1], t[0]))
    restore_writer_weights(model, snapshot)
    direction = directions[chosen_layer].to(device=device, dtype=dtype)
    orthogonalize_model(model, direction, alpha)
    edit_s = time.time() - t0
    n_mats = 1 + 2 * n_layers
    detail = (f"— orthogonalised embed + {n_layers}x(o_proj, down_proj) in {edit_s:.1f}s "
              f"({n_mats} matrices), no training, no hooks")
    print(f"-> kept layer {green(bold(str(chosen_layer)))} at α = {green(bold(str(alpha)))}  {dim(detail)}")

    # ---- 7. after ----------------------------------------------------------
    banner("7 · After: does the edited model still refuse?")
    abl_outputs = generate(model, tokenizer, harmful_test, device, args.max_new_tokens, args.batch_size)
    abl_rate = refusal_rate(abl_outputs)
    abl_answered = on_topic_rate(harmful_test, abl_outputs)
    abl_benign = generate(model, tokenizer, [BENIGN_SANITY_PROMPT], device, args.max_new_tokens, 1)[0]
    print(f"refusal rate: {red(f'{base_rate:.0%}')}  ->  {green(bold(f'{abl_rate:.0%}'))}   (n={len(harmful_test)})")
    print(f"real on-topic answers: {base_answered:.0%}  ->  {bold(f'{abl_answered:.0%}')}  {dim('(the honest number: not refusing is not the same as answering)')}")
    print(f"{dim('harmless check')} \"{BENIGN_SANITY_PROMPT}\"\n  → {abl_benign.strip()[:200]}")

    if args.show_outputs:
        banner("Held-out completions (truncated) — you asked for these with --show-outputs")
        for p, b, a in zip(harmful_test, base_outputs, abl_outputs):
            print(f"\nPROMPT   : {p[:80]}")
            print(f"BEFORE   : [{'refusal' if is_refusal(b) else 'answer'}] {b.strip()[:120]!r}")
            print(f"AFTER    : [{'refusal' if is_refusal(a) else 'answer'}] {a.strip()[:120]!r}")

    banner("Done")
    print(f"model          : {args.model}")
    print(f"edit           : layer {chosen_layer} / {n_layers}, α = {alpha}")
    print(f"refusal rate   : {red(f'{base_rate:.0%}')} -> {green(bold(f'{abl_rate:.0%}'))}")
    print(f"on-topic answer: {base_answered:.0%} -> {abl_answered:.0%}")
    print(f"total time     : {time.time() - t_start:.1f}s")
    print(dim("note           : this removes explicit refusals; it does not add capability, and usually"))
    print(dim("                 costs a little general quality (heal it: python heal.py)."))

    if not args.no_save:
        args.save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print(f"\nsaved the edited model to {args.save}")
        print(dim("chat with it:    python chat.py --compare      (/q to quit)"))


if __name__ == "__main__":
    raise SystemExit(main())
