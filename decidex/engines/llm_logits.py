"""LLM direct-logit engine: the SemIf (formerly OpenJev) approach.

A frozen causal LM (default Qwen3-4B) scores every option in a single forward
pass by reading the logits of letter labels (A, B, C, ...) at the last
position. No tokens are sampled and nothing is generated, which is what makes
this a "System One" readout: probabilities come out directly and no output
tokens are spent. This reproduces the *interface pattern* of Jev with an open
model, not Jev's undisclosed architecture or RLCD training.

Reference: https://github.com/TheoLeeCJ/SemIf (direct typed logits, 0 output
tokens, 5.2x faster than autoregressive JSON on identical questions).

Design points:

- All questions in one request share the same rendered state, so they are
  batched into padded forward passes (left padding keeps each row's final
  position aligned for the label readout).
- Up to 26 options read out directly via letter labels. Beyond that (the
  official API allows 255), options switch to independent relevance probes
  ("does this option apply? yes/no") whose scores are renormalized -- the same
  two-stage shape TypeSafe describes for high-cardinality choices.
- A temperature (default 1.0) softens or sharpens the raw logit distribution;
  this is the calibration knob that stands in for Jev's RLCD training.
"""

from __future__ import annotations

import string
from collections import OrderedDict

from decidex.calib import softmax
from decidex.engines.base import Engine

LETTERS = string.ascii_uppercase
DIRECT_LABEL_LIMIT = 26  # single-letter readout capacity
CHUNK = 16  # max questions per padded forward pass (bounds KV memory);
             # measured 20% faster than 8 on a 10-question fan-out, flat beyond


class LLMLogitsEngine(Engine):
    name = "llm"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        temperature: float = 1.0,
        dtype: str = "auto",
        max_input_tokens: int = 8192,
        prefix_reuse: bool = True,
        lora_path: str | None = None,
    ):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or os.environ.get("DECIDEX_MODEL", "Qwen/Qwen3-4B")
        super().__init__(model_id=model_name, temperature=temperature)
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        # Optional LoRA adapter (e.g. distilled from official outputs; see
        # benchmarks/distill_train.py). Loaded on top of the frozen base.
        self.lora_path = lora_path or os.environ.get("DECIDEX_LORA") or None
        # KV prefix reuse: one forward over the shared state, then every
        # question (and every relevance probe) continues from that cache.
        # SemIf measured this family of tricks as the biggest throughput win
        # (2.3x -> 20 decisions/s on their workload). Disable with
        # DECIDEX_PREFIX_REUSE=0.
        self.prefix_reuse = prefix_reuse and os.environ.get("DECIDEX_PREFIX_REUSE", "1") != "0"

        # Cross-request LRU of state-prefix KV caches: repeated queries over
        # the same long state skip the prefill forward entirely. Budget in GB
        # (DECIDEX_PREFIX_CACHE_GB, default 4; 0 disables).
        self._prefix_cache: OrderedDict[str, tuple] = OrderedDict()  # prefix -> (past, prefix_len)
        self._prefix_cache_budget_gb = float(os.environ.get("DECIDEX_PREFIX_CACHE_GB", "4"))
        self._prefix_cache_bytes = 0
        # A/B knobs: prompt_variant in {"plain", "strict", "chat"};
        # ensemble_rounds > 1 averages distributions over option-order
        # permutations (self-consistency), trading latency for accuracy.
        self.prompt_variant = "plain"
        self.ensemble_rounds = 1

        if device is None:
            device = os.environ.get("DECIDEX_DEVICE")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs: dict = {}
        if dtype in ("int8", "int4"):
            # bitsandbytes quantization: the only way 7-8B models fit in 16GB.
            # Quantization perturbs logits; the distillation pipeline is the
            # documented way to recover fidelity on top.
            from transformers import BitsAndBytesConfig

            if device == "cpu":
                raise ValueError(f"dtype={dtype!r} requires a CUDA device")
            if dtype == "int8":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
        else:
            load_kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
        if device != "cpu":
            load_kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if self.lora_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            print(f"[decidex] LoRA adapter loaded: {self.lora_path}")
        else:
            self.model.eval()

        # Letter label -> candidate token ids (label with and without a leading
        # space; the max logit over variants is the label's score).
        self._label_ids: dict[str, list[int]] = {}
        for letter in LETTERS:
            ids = set()
            for variant in (letter, " " + letter):
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if encoded:
                    ids.add(encoded[0])
            if not ids:
                raise RuntimeError(
                    f"tokenizer produced no token for label {letter!r}; "
                    "this model cannot be used for letter readout"
                )
            self._label_ids[letter] = sorted(ids)

    # -- tokenizer passthrough ---------------------------------------------
    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # -- prompt construction -------------------------------------------------
    SYSTEM_PREAMBLE = (
        "You are a precise decision engine. Evaluate the STATE against the "
        "QUESTION and pick the single best option."
    )

    def build_prefix(self, state_text: str) -> str:
        """The state-bearing part of the prompt, shared by every question."""
        return f"{self.SYSTEM_PREAMBLE}\n\nSTATE:\n{state_text}\n"

    _SUFFIX_ENDINGS = {
        "plain": ("Answer with the letter of the single best option.", "Answer:"),
        "strict": ("Respond with ONLY the letter of the single best option.", "Answer:"),
    }

    def build_suffix(self, instruction_text: str, option_texts: list[str]) -> str:
        """The question-bearing part; concatenated after the state prefix."""
        lines = ["", "QUESTION:", instruction_text, "", "OPTIONS:"]
        for letter, option in zip(LETTERS, option_texts):
            lines.append(f"{letter}. {option}")
        ending, cue = self._SUFFIX_ENDINGS.get(self.prompt_variant,
                                               self._SUFFIX_ENDINGS["plain"])
        lines.extend(["", ending, cue])
        return "\n".join(lines)

    def build_prompt(self, state_text: str, instruction_text: str, option_texts: list[str]) -> str:
        prompt = self.build_prefix(state_text) + self.build_suffix(instruction_text, option_texts)
        if self.prompt_variant == "chat":
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    # -- scoring --------------------------------------------------------------
    def score_batch(self, jobs: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        """Score jobs, writing each result at its job's position.

        Jobs sharing the same state are grouped: the state prefix is
        forwarded once and every question suffix (direct letter readout or
        relevance probe) continues from that KV cache. Results are placed by
        job index so any request stays aligned.
        """
        results: list[list[float] | None] = [None] * len(jobs)
        by_state: dict[str, list[int]] = {}
        for idx, (state, _instruction, _options) in enumerate(jobs):
            by_state.setdefault(state, []).append(idx)

        for state, indices in by_state.items():
            suffix_jobs = self._build_suffix_jobs(jobs, indices)
            # The chat template wraps the whole prompt, so prefix reuse only
            # applies to the plain/strict completion variants.
            if self.prefix_reuse and self.prompt_variant != "chat" and len(suffix_jobs) >= 2:
                try:
                    self._score_with_prefix_reuse(state, suffix_jobs, results)
                    continue
                except Exception as err:  # noqa: BLE001
                    # Cache APIs vary across transformers versions; any
                    # mismatch falls back to the always-correct full prompts.
                    print(f"[decidex] prefix reuse unavailable ({err}); using full prompts")
            self._score_full_prompts(state, suffix_jobs, results)

        if any(r is None for r in results):
            missing = [i for i, r in enumerate(results) if r is None]
            raise RuntimeError(f"engine left jobs unscored: {missing}")
        return results  # type: ignore[return-value]

    # suffix job: (job index, suffix text, n_options) where n_options > 0
    # means direct letter readout over that many options, and 0 means a
    # two-option relevance probe read out as A-vs-B.
    def _build_suffix_jobs(self, jobs, indices) -> list[tuple[int, str, int]]:
        suffix_jobs: list[tuple[int, str, int]] = []
        for idx in indices:
            _state, instruction, options = jobs[idx]
            if len(options) <= DIRECT_LABEL_LIMIT:
                suffix_jobs.append(
                    (idx, self.build_suffix(instruction, options), len(options))
                )
            else:
                probe_instruction = (
                    f"{instruction}\n(One candidate option is being considered on its own.)"
                )
                for option in options:
                    suffix_jobs.append(
                        (
                            idx,
                            self.build_suffix(
                                probe_instruction,
                                [f"yes - {option}", "no - this option does not apply"],
                            ),
                            0,
                        )
                    )
        return suffix_jobs

    def _readout(self, row, n_options: int) -> list[float]:
        """Turn a last-position logit row into a distribution."""
        if n_options > 0:
            scores = [
                max(row[t].item() for t in self._label_ids[LETTERS[k]])
                for k in range(n_options)
            ]
        else:  # relevance probe: option's score is the A-vs-B letter gap
            scores = [
                max(row[t].item() for t in self._label_ids["A"])
                - max(row[t].item() for t in self._label_ids["B"])
            ]
        return scores

    def _store(self, results, suffix_jobs, rows, probe_scores: dict[int, list[float]]) -> None:
        """Fold one batch of rows into results; probe scores accumulate across
        batches and are normalized by the caller once every probe has run."""
        for row, (idx, _suffix, n_options) in zip(rows, suffix_jobs):
            scores = self._readout(row, n_options)
            if n_options > 0:
                results[idx] = softmax(scores, temperature=self.temperature)
            else:
                probe_scores.setdefault(idx, []).extend(scores)

    def _finalize_probes(self, results, probe_scores: dict[int, list[float]]) -> None:
        for idx, scores in probe_scores.items():
            results[idx] = softmax(scores, temperature=self.temperature)

    def _score_full_prompts(self, state: str, suffix_jobs, results) -> None:
        probe_scores: dict[int, list[float]] = {}
        if self.prompt_variant == "chat":
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": self.build_prefix(state) + suffix}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _idx, suffix, _n in suffix_jobs
            ]
        else:
            prompts = [self.build_prefix(state) + suffix for _idx, suffix, _n in suffix_jobs]
        rows = self._rows_with_backoff(prompts)
        self._store(results, suffix_jobs, rows, probe_scores)
        self._finalize_probes(results, probe_scores)

    def score(self, state_text: str, instruction_text: str, option_texts: list[str]) -> list[float]:
        """Score one job, optionally ensembling over option-order permutations.

        Permuting option order and averaging the distributions removes the
        model's position bias; accuracy typically improves slightly at
        ensemble_rounds times the latency.
        """
        if self.ensemble_rounds <= 1 or len(option_texts) < 3:
            return super().score(state_text, instruction_text, option_texts)
        import random

        n = len(option_texts)
        rng = random.Random(0x5EED)
        permutations = [list(range(n))]
        for _ in range(self.ensemble_rounds - 1):
            order = list(range(n))
            rng.shuffle(order)
            permutations.append(order)
        jobs = [
            (state_text, instruction_text, [option_texts[i] for i in order])
            for order in permutations
        ]
        distributions = self.score_batch(jobs)
        averaged = [0.0] * n
        for order, dist in zip(permutations, distributions):
            for slot, prob in zip(order, dist):
                averaged[slot] += prob / len(permutations)
        return averaged

    def _rows_with_backoff(self, prompts: list[str]):
        """Forward prompts in chunks, halving the chunk on CUDA OOM.

        Long states make attention memory quadratic in batch, so a request
        that fits at batch 1 may not fit at batch 8; backing off keeps the
        server alive instead of crashing the process.
        """
        torch = self.torch
        rows = []
        size = min(CHUNK, len(prompts))
        i = 0
        while i < len(prompts):
            batch = prompts[i : i + size]
            try:
                rows.extend(self._forward_last_logits(batch))
                i += size
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"[decidex] CUDA OOM; retrying with batch size {size}")
        return rows

    def _score_with_prefix_reuse(self, state: str, suffix_jobs, results) -> None:
        import copy

        torch = self.torch
        past_template, prefix_len = self._get_prefix_cache(self.build_prefix(state))
        probe_scores: dict[int, list[float]] = {}
        size = min(CHUNK, len(suffix_jobs))
        start = 0
        while start < len(suffix_jobs):
            chunk = suffix_jobs[start : start + size]
            try:
                batch = len(chunk)
                past = copy.deepcopy(past_template)
                if batch > 1:
                    past.batch_repeat_interleave(batch)

                original_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                try:
                    encoded = self.tokenizer(
                        [suffix for _idx, suffix, _n in chunk], return_tensors="pt", padding=True
                    )
                finally:
                    self.tokenizer.padding_side = original_side

                input_ids = encoded["input_ids"].to(self.model.device)
                suffix_mask = encoded["attention_mask"].to(input_ids.device)
                ones = torch.ones(
                    (batch, prefix_len), dtype=suffix_mask.dtype, device=input_ids.device
                )
                attention_mask = torch.cat([ones, suffix_mask], dim=1)
                # Positions cover only the new tokens; pads clamp to a harmless
                # dummy position since they are attention-masked out.
                position_ids = (prefix_len + (suffix_mask.cumsum(-1) - 1)).clamp(min=0)
                with torch.inference_mode():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past,
                    )
                rows = output.logits[:, -1, :].float().cpu().unbind(0)
                self._store(results, chunk, rows, probe_scores)
                start += size
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"[decidex] CUDA OOM; retrying with batch size {size}")
        self._finalize_probes(results, probe_scores)

    def _get_prefix_cache(self, prefix_text: str) -> tuple:
        """Return (past_kv_template, prefix_len), prefilled once per state.

        Hits come from an LRU keyed by the exact prefix text, bounded by a
        KV-byte budget so long states cannot exhaust GPU memory.
        """
        cached = self._prefix_cache.get(prefix_text)
        if cached is not None:
            self._prefix_cache.move_to_end(prefix_text)  # type: ignore[attr-defined]
            return cached
        past, prefix_len = self._forward_prefix(prefix_text)
        if self._prefix_cache_budget_gb > 0:
            entry_bytes = self._kv_bytes_for(prefix_len)
            budget = self._prefix_cache_budget_gb * 1e9
            while self._prefix_cache and self._prefix_cache_bytes + entry_bytes > budget:
                evicted_text, evicted = next(iter(self._prefix_cache.items()))
                del self._prefix_cache[evicted_text]
                self._prefix_cache_bytes -= self._kv_bytes_for(evicted[1])
            if entry_bytes <= budget:
                self._prefix_cache[prefix_text] = (past, prefix_len)
                self._prefix_cache_bytes += entry_bytes
        return past, prefix_len

    def _kv_bytes_for(self, seq_len: int) -> int:
        config = self.model.config
        layers = getattr(config, "num_hidden_layers", 1)
        kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 1))
        head_dim = getattr(config, "head_dim", None) or (
            getattr(config, "hidden_size", 1) // getattr(config, "num_attention_heads", 1)
        )
        element_size = next(self.model.parameters()).element_size()
        return seq_len * layers * 2 * kv_heads * head_dim * element_size

    def _forward_prefix(self, prefix_text: str):
        torch = self.torch
        encoded = self.tokenizer(prefix_text, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded, use_cache=True)
        return output.past_key_values, encoded["input_ids"].shape[1]

    def _forward_last_logits(self, prompts: list[str]):
        """One padded forward pass; returns last-position logit rows (on CPU).

        Prompts are left-padded so each row's final position aligns. Explicit
        position_ids are derived from the attention mask: this keeps real
        tokens at 0..n positions per row, which matters for models with
        absolute (learned) position embeddings and is harmless for RoPE.
        """
        torch = self.torch
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = original_side
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        attention_mask = encoded["attention_mask"]
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            output = self.model(**encoded, position_ids=position_ids)
        return output.logits[:, -1, :].float().cpu().unbind(0)
