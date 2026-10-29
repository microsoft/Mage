"""Attention backend shim — switchable between Flash Attention 2 and 4.

Exports a single ``flash_attn_varlen_func`` with the FA2 calling convention.
The underlying kernel is selected at runtime via ``set_attn_backend(name)``
(default: ``"flash2"``). The selected kernel is resolved lazily on the first
call so model-config-driven selection (which happens after this module is
imported) takes effect.

Modules that previously did ``from flash_attn import flash_attn_varlen_func``
should import from here instead.

For the FA4 path, calling-convention differences are normalised:

* ``window_size=(-1, -1)`` (FA2 "no window") -> ``(None, None)`` (FA4).
* ``block_table`` -> ``page_table``.
* FA4's optional ``(out, lse)`` tuple return is unwrapped to ``out``.
* ``dropout_p>0`` / ``alibi_slopes`` / ``return_attn_probs`` raise on FA4.
"""

from __future__ import annotations

from typing import Any, Callable

_FA2_ALIASES = {"flash2", "fa2", "flash_attention_2", "flash_attn_2"}
_FA4_ALIASES = {"flash4", "fa4", "flash_attention_4", "flash_attn_4"}
_SDPA_ALIASES = {"sdpa", "torch_sdpa", "scaled_dot_product_attention"}

_BACKEND: str = "flash2"
_RESOLVED_FN: Callable[..., Any] | None = None


def _normalize(name: str) -> str:
    n = name.lower().strip()
    if n in _FA2_ALIASES:
        return "flash2"
    if n in _FA4_ALIASES:
        return "flash4"
    if n in _SDPA_ALIASES:
        return "sdpa"
    raise ValueError(
        f"Unknown attention backend {name!r}; expected one of "
        f"{sorted(_FA2_ALIASES | _FA4_ALIASES | _SDPA_ALIASES)}"
    )


def set_attn_backend(name: str) -> None:
    """Select the flash-attn backend used by ``flash_attn_varlen_func``.

    Safe to call multiple times; clears the cached resolution on change.
    """
    global _BACKEND, _RESOLVED_FN
    new = _normalize(name)
    if new != _BACKEND:
        _RESOLVED_FN = None
    _BACKEND = new



def _resolve_fa2() -> Callable[..., Any]:
    from flash_attn import flash_attn_varlen_func as _fn
    return _fn


def _resolve_fa4() -> Callable[..., Any]:
    import torch
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            from loguru import logger
            logger.warning("FlashAttention-4 requires SM90+ (Hopper). Falling back to FA2.")
            return _resolve_fa2()

    from flash_attn.cute import flash_attn_varlen_func as _fa4_fn

    def _fa4_wrapper(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p: float = 0.0,
        softmax_scale=None,
        causal: bool = False,
        window_size=(-1, -1),
        softcap: float = 0.0,
        alibi_slopes=None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        block_table=None,
        **_unused: Any,
    ):
        if dropout_p and dropout_p > 0:
            raise NotImplementedError("FA4 backend does not support dropout_p>0")
        if alibi_slopes is not None:
            raise NotImplementedError("FA4 backend does not support alibi_slopes")
        if return_attn_probs:
            raise NotImplementedError("FA4 backend does not support return_attn_probs")

        win_l, win_r = window_size
        if win_l == -1:
            win_l = None
        if win_r == -1:
            win_r = None

        out = _fa4_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(win_l, win_r),
            softcap=softcap,
            deterministic=deterministic,
            page_table=block_table,
            return_lse=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    return _fa4_wrapper


def _resolve_sdpa() -> Callable[..., Any]:
    """FA2 varlen → per-sequence torch.SDPA fallback.

    Use when flash-attn is unavailable (e.g. CUDA 13 has no prebuilt wheel
    and source build is brittle). Slower than FA2 (one SDPA dispatch per
    sequence), but functionally equivalent for the dense / causal / no-alibi
    paths mageflow actually uses. Window / softcap / alibi / paged-attn /
    return_attn_probs are not supported and will raise.
    """
    import torch
    import torch.nn.functional as F

    def _sdpa_wrapper(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p: float = 0.0,
        softmax_scale=None,
        causal: bool = False,
        window_size=(-1, -1),
        softcap: float = 0.0,
        alibi_slopes=None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        block_table=None,
        **_unused: Any,
    ):
        if dropout_p and dropout_p > 0:
            raise NotImplementedError("SDPA backend does not support dropout_p>0")
        if alibi_slopes is not None:
            raise NotImplementedError("SDPA backend does not support alibi_slopes")
        if return_attn_probs:
            raise NotImplementedError("SDPA backend does not support return_attn_probs")
        if softcap and softcap > 0:
            raise NotImplementedError("SDPA backend does not support softcap")
        if window_size not in ((-1, -1), (None, None), (0, 0)):
            raise NotImplementedError(
                f"SDPA backend does not support sliding window (got {window_size})"
            )
        if block_table is not None:
            raise NotImplementedError("SDPA backend does not support paged attention")
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("SDPA backend requires cu_seqlens_q and cu_seqlens_k")

        # GQA: FA2 broadcasts k/v across query head groups natively; torch SDPA
        # does not (the q vs k head-dim mismatch is the AssertionError "tensor
        # a (32) must match tensor b (8) at non-singleton dimension 1" we'd see
        # otherwise). Repeat k/v along the head dim to match q before the loop.
        n_heads_q = q.shape[1]
        n_heads_kv = k.shape[1]
        if n_heads_q != n_heads_kv:
            if n_heads_q % n_heads_kv != 0:
                raise ValueError(
                    f"SDPA backend GQA expansion requires q heads ({n_heads_q}) "
                    f"to be divisible by k/v heads ({n_heads_kv})"
                )
            repeat = n_heads_q // n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # q/k/v: (total_tokens, nheads, head_dim). Batched pad sequence SDPA
        # to avoid sequential python loop overhead.
        cu_q = cu_seqlens_q.tolist()
        cu_k = cu_seqlens_k.tolist()
        
        q_list = [q[qs:qe] for qs, qe in zip(cu_q[:-1], cu_q[1:])]
        k_list = [k[ks:ke] for ks, ke in zip(cu_k[:-1], cu_k[1:])]
        v_list = [v[ks:ke] for ks, ke in zip(cu_k[:-1], cu_k[1:])]
        
        q_pad = torch.nn.utils.rnn.pad_sequence(q_list, batch_first=True) # (B, max_Lq, H, D)
        k_pad = torch.nn.utils.rnn.pad_sequence(k_list, batch_first=True)
        v_pad = torch.nn.utils.rnn.pad_sequence(v_list, batch_first=True)
        
        q_pad = q_pad.transpose(1, 2) # (B, H, max_Lq, D)
        k_pad = k_pad.transpose(1, 2)
        v_pad = v_pad.transpose(1, 2)
        
        B = len(q_list)
        max_q = q_pad.shape[2]
        max_k = k_pad.shape[2]
        attn_mask = torch.zeros(B, max_q, max_k, dtype=torch.bool, device=q.device)
        for i, (lq, lk) in enumerate(zip([len(x) for x in q_list], [len(x) for x in k_list])):
            attn_mask[i, :lq, :lk] = True
        attn_mask = attn_mask.unsqueeze(1)
        
        out_pad = F.scaled_dot_product_attention(
            q_pad, k_pad, v_pad,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        
        out_pad = out_pad.transpose(1, 2) # (B, max_q, H, D)
        
        outs = []
        for i, lq in enumerate([len(x) for x in q_list]):
            outs.append(out_pad[i, :lq])
            
        return torch.cat(outs, dim=0).contiguous()

    return _sdpa_wrapper


def _resolve() -> Callable[..., Any]:
    global _RESOLVED_FN
    if _RESOLVED_FN is None:
        if _BACKEND == "flash4":
            _RESOLVED_FN = _resolve_fa4()
        elif _BACKEND == "sdpa":
            _RESOLVED_FN = _resolve_sdpa()
        else:
            _RESOLVED_FN = _resolve_fa2()
    return _RESOLVED_FN


def flash_attn_varlen_func(*args, **kwargs):
    return _resolve()(*args, **kwargs)


__all__ = ["flash_attn_varlen_func", "set_attn_backend"]
