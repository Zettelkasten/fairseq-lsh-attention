# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq.modules import lsh_attention

try:
    from xformers.components.attention import build_attention
    from xformers.components.attention.utils import maybe_merge_masks

    _xformers_available = True
except ImportError:
    _xformers_available = False

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout


@with_incremental_state
class MultiheadVanillaAttention(nn.Module):
    """Just like MultiheadAttention, but implemented in a simpler way. Easier for debugging.
    Also see MultiheadLshAttention.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        *,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # ignored
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # ignored
    ):
        super().__init__()

        xformers_att_config = utils.eval_str_dict(xformers_att_config)
        assert xformers_att_config is None, "Not implemented"
        assert q_noise == 0, "not implemented"
        assert not add_bias_kv, "Not implemented"
        assert not add_zero_attn, "Not implemented"
        assert self_attention != encoder_decoder_attention, "need exactly one"

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.query_dim= self.head_dim
        self.key_dim = self.head_dim
        self.value_dim = self.head_dim
        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.scaling = self.key_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.key_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.value_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.key_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.value_dim, self.embed_dim, bias=bias)

        self.beam_size = 1
        self.reset_parameters()

    def reset_parameters(self):
        if self.key_dim == self.value_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert query is not None and key is not None and value is not None, "Not implemented"

        num_queries, num_batch, _ = query.size()
        num_keys, _, _ = key.size()
        assert tuple(query.size()) == (num_queries, num_batch, self.embed_dim)
        assert tuple(key.size()) == (num_keys, num_batch, self.embed_dim)
        assert tuple(value.size()) == (num_keys, num_batch, self.embed_dim)

        q: torch.Tensor = self.q_proj(query)  # [query-time, batch, head * key-dim]
        k: torch.Tensor = self.k_proj(key)  # [key-time, batch, head * key-dim]
        v: torch.Tensor = self.v_proj(value)  # [key-time, batch, head * value-dim]

        q *= self.scaling

        q = q.view(num_queries, num_batch, self.num_heads, self.key_dim)
        k = k.view(num_keys, num_batch, self.num_heads, self.key_dim)
        v = v.view(num_keys, num_batch, self.num_heads, self.value_dim)

        energy: torch.Tensor = torch.einsum("ibnf,jbnf->bnij", q, k)  # (num_batch, self.num_heads, num_queries, num_keys)

        if attn_mask is not None:
            # Warning: We do not want to use attn_mask itself, as this requires O(T^2) memory itself.
            # Instead, we just assume that it is used for causal masking and nothing else.
            causal_mask = torch.triu(float("-inf") * energy.new_ones(1, 1, num_queries, num_keys), diagonal=1)
            energy += causal_mask

        if key_padding_mask is not None:
            assert tuple(key_padding_mask.size()) == (num_batch, num_keys)
            key_mask = torch.where(key_padding_mask.view(num_batch, 1, 1, num_keys), float("-inf"), 0.0)
            energy += key_mask

        weights = torch.softmax(energy, dim=-1)  # (num_batch, self.num_heads, num_queries, num_keys)
        dropped_weights = self.dropout_module(weights)

        head_context = torch.einsum("bnij,jbnf->ibnf", dropped_weights, v)  # (num_queries, num_batch, self.num_heads, self.value_dim)  # noqa
        context = head_context.reshape(num_queries, num_batch, self.num_heads * self.value_dim)
        out = self.out_proj(context)

        need_weights = need_weights or need_head_weights
        if need_weights:
            out_weights = weights.transpose(1, 0)  # (self.num_heads, num_batch, num_queries, num_keys)
            if not need_head_weights:
                # average over head dim
                out_weights = out_weights.mean(dim=0)  # (num_batch, num_queries, num_keys)
        else:
            out_weights = None

        out_weights = locals()

        return out, out_weights

    def set_beam_size(self, beam_size):
        """Used for effiecient beamable enc-dec attention"""
        self.beam_size = beam_size

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
