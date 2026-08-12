import ctypes
import json
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import numpy as np

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta


def _read_safetensors(path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (name, raw_bytes_view) for every tensor in a safetensors file.

    The safetensors layout is: 8-byte little-endian header length, a JSON
    header mapping names to {dtype, shape, data_offsets}, then raw data.
    Reading the raw bytes directly avoids the numpy bfloat16 dtype support
    that the safetensors numpy backend requires.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if info["dtype"] != "BF16":
                raise TypeError(f"unsupported weight dtype: {info['dtype']} ({name})")
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            yield name, np.frombuffer(f.read(end - start), dtype=np.uint8)


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        dh = config.get("head_dim", hs // nh)
        maxseq = min(config.get("max_position_embeddings", 2048), 2048)

        meta = LlaisysQwen2Meta(
            dtype=int(DataType.BF16),
            nlayer=config["num_hidden_layers"],
            hs=hs,
            nh=nh,
            nkvh=config["num_key_value_heads"],
            dh=dh,
            di=config["intermediate_size"],
            maxseq=maxseq,
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config.get("eos_token_id", 0),
        )

        self._device = device
        self._end_token = meta.end_token
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), None, 0
        )
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        expected = set()
        loaded = set()
        for file in sorted(model_path.glob("*.safetensors")):
            for name_, raw in _read_safetensors(file):
                expected.add(name_)
                tensor = self._map_weight(weights, name_)
                if tensor is None:
                    raise RuntimeError(f"Unknown weight name: {name_}")
                LIB_LLAISYS.tensorLoad(
                    tensor, raw.ctypes.data_as(ctypes.c_void_p)
                )
                loaded.add(name_)

        missing = expected - loaded
        if missing:
            raise RuntimeError(f"Failed to load weights: {sorted(missing)}")

    def __del__(self):
        if getattr(self, "_model", None) is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    @staticmethod
    def _map_weight(weights, name: str):
        """Map a safetensors name to its llaisysTensor_t slot in LlaisysQwen2Weights."""
        if name == "model.embed_tokens.weight":
            return weights.in_embed
        if name == "lm_head.weight":
            return weights.out_embed
        if name == "model.norm.weight":
            return weights.out_norm_w

        parts = name.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
            return None
        layer = int(parts[2])
        field = {
            ("input_layernorm", "weight"): "attn_norm_w",
            ("post_attention_layernorm", "weight"): "mlp_norm_w",
            ("self_attn", "q_proj", "weight"): "attn_q_w",
            ("self_attn", "q_proj", "bias"): "attn_q_b",
            ("self_attn", "k_proj", "weight"): "attn_k_w",
            ("self_attn", "k_proj", "bias"): "attn_k_b",
            ("self_attn", "v_proj", "weight"): "attn_v_w",
            ("self_attn", "v_proj", "bias"): "attn_v_b",
            ("self_attn", "o_proj", "weight"): "attn_o_w",
            ("mlp", "gate_proj", "weight"): "mlp_gate_w",
            ("mlp", "up_proj", "weight"): "mlp_up_w",
            ("mlp", "down_proj", "weight"): "mlp_down_w",
        }.get(tuple(parts[3:]))
        if field is None:
            return None
        return getattr(weights, field)[layer]

    def _infer(self, token_ids: Sequence[int]) -> int:
        arr = (ctypes.c_int64 * len(token_ids))(*token_ids)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(token_ids))
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # The assignment requires argmax sampling, so top_k/top_p/temperature
        # are ignored (the test passes 1.0 / 1 / 1.0, i.e. greedy).
        if max_new_tokens is None:
            max_new_tokens = 128

        outputs = list(inputs)
        n_prompt = len(inputs)

        # Prefill the whole prompt in one forward pass, then decode one token
        # at a time (the C++ side keeps the KV cache and position state).
        nxt = self._infer(inputs)
        outputs.append(nxt)
        while nxt != self._end_token and len(outputs) - n_prompt < max_new_tokens:
            nxt = self._infer([nxt])
            outputs.append(nxt)

        return outputs
