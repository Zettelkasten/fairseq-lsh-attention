# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple, Sequence

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
from fairseq.modules.quant_noise import quant_noise


# TODO: move this into xformers?
# TODO: uint8 input type should just output a bool
def _mask_for_xformers(mask: Tensor, to_dtype: Optional[torch.dtype] = None):
    """
    call to pytorch multihead accepts three mask types:
        - ByteTensor where non-zero means to mask
        - FloatTensor which is an additive mask
        - BoolTensor where True means to mask
    xFormers currently accepts boolean and additive maks. For boolean masks
    the values have opposite meaning. For a BoolTensor True mean to keep the value.
    """
    float_types = [torch.float, torch.float16]
    # If an input mask is a float it is an additive mask. Otherwise it is either uint8 or bool.
    additive = mask.dtype in float_types
    # If to_dype is not specified, keep same dtype as mask.
    to_dtype = mask.dtype if to_dtype is None else to_dtype
    to_additive = to_dtype in float_types

    if additive:
        if to_additive:
            return mask.to(to_dtype)
        mask = mask < 0

    if to_additive:
        # return additive mask
        new_mask = torch.zeros_like(mask, dtype=to_dtype)
        new_mask = new_mask.masked_fill_(mask, -float("inf"))
        return new_mask

    # In xFormers True is value to keep rather than value to mask
    mask = ~mask.to(torch.bool)
    mask = mask.to(to_dtype)
    return mask


@with_incremental_state
class MultiheadLshAttention(nn.Module):
    """Multi-headed locality-sensitive hashing attention.

    See "Locality-Sensitive Hashing for Long Context Neural Machine Translation" (Petrick et al., 2022)
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
        share_kq: Optional[bool] = None,
        num_rounds: int,
        num_hashes: int,
        chunk_size: int,
        mask_different_hashes: bool = True,
        mask_current: Optional[bool] = None,
        mask_different_rounds: bool = True
    ):
        super().__init__()

        xformers_att_config = utils.eval_str_dict(xformers_att_config)
        assert xformers_att_config is None, "Not implemented"
        assert q_noise == 0, "not implemented"
        assert not add_bias_kv, "Not implemented"
        assert not add_zero_attn, "Not implemented"
        assert self_attention != encoder_decoder_attention, "need exactly one"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.query_dim= self.head_dim
        self.key_dim = self.head_dim
        self.value_dim = self.head_dim
        self.num_heads = num_heads

        self.num_rounds = num_rounds
        assert num_hashes % 2 == 0, "num_hashes must be divisible by 2"
        self.num_hashes = num_hashes
        self.chunk_size = chunk_size
        self.mask_different_hashes = mask_different_hashes
        if mask_current is None:
            mask_current = self.self_attention and self.mask_different_hashes
        assert not (self.encoder_decoder_attention and mask_current)
        self.mask_current = mask_current
        self.mask_different_rounds = mask_different_rounds

        if share_kq is None:
            share_kq = self_attention
        assert not (share_kq and encoder_decoder_attention)
        self.share_kq = share_kq
        assert not self.share_kq or self.self_attention, "Can only share keys=queries in self-attention"

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.scaling = self.key_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.key_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.value_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.key_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.value_dim, self.embed_dim, bias=bias)

        self.hash_proj_weight = nn.Parameter(torch.empty(self.num_rounds, self.num_heads, self.key_dim, self.num_hashes // 2))

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

        nn.init.xavier_uniform_(self.hash_proj_weight)

    @staticmethod
    def _ceildiv(a, b):
        return -(a // -b)

    @staticmethod
    def count_duplicates(dim, index: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
        src = index.new_ones(*out_shape)
        out = index.new_zeros(*out_shape)
        if index.size(dim) <= src.size(dim):
            return out.scatter(dim=dim, index=index, src=src, reduce="add")
        # torch does not handle this case currently, see https://github.com/pytorch/pytorch/issues/63265
        # this is a workaround for that
        padding = index.size(dim) - src.size(dim)
        src_extended = F.pad(src, (0, padding) + (0, 0) * (src.ndim - dim - 1), value=1)
        scattered = out.scatter(dim=dim, index=index, src=src_extended, reduce="add")
        return scattered[(slice(None, None),) * dim + (slice(None, src.size(dim)),) + (slice(None, None),) * (src.ndim - dim - 1)]

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
        override_hashes: Optional[torch.Tensor] = None,
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
        assert not before_softmax, "Not implemented"
        del static_kv  # ignored

        num_queries, num_batch, _ = query.size()
        num_keys, _, _ = key.size()
        assert tuple(query.size()) == (num_queries, num_batch, self.embed_dim)
        assert tuple(key.size()) == (num_keys, num_batch, self.embed_dim)
        assert tuple(value.size()) == (num_keys, num_batch, self.embed_dim)

        # Warning: We do not want to use attn_mask itself, as this requires O(T^2) memory itself.
        # Instead, we just assume that it is used for causal masking and nothing else.
        causal = attn_mask is not None
        num_chunk_offsets = 2 if causal else 3

        q: torch.Tensor = self.q_proj(query)  # [query-time, batch, head * key-dim]
        if self.share_kq:
            assert num_queries == num_keys
            k = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)  # [key-time, batch, head * key-dim]
        else:
            k: torch.Tensor = self.k_proj(key)  # [key-time, batch, head * key-dim]
        v: torch.Tensor = self.v_proj(value)  # [key-time, batch, head * value-dim]

        q = q * self.scaling

        q = q.view(num_queries, num_batch, self.num_heads, self.key_dim)
        k = k.view(num_keys, num_batch, self.num_heads, self.key_dim)
        v = v.view(num_keys, num_batch, self.num_heads, self.value_dim)

        def apply_hash(seq: torch.Tensor, seq_padding_mask: Optional[torch.Tensor]):
            num_frames = seq.size(0)
            assert tuple(seq.size()) == (num_frames, num_batch, self.num_heads, self.key_dim)
            assert seq_padding_mask is None or tuple(seq_padding_mask.size()) == (num_batch, num_frames)
            linear = torch.einsum("ibnf,rnfh->ibrnh", seq, self.hash_proj_weight)  # (num_queries, num_batch, num_round, num_head, hash_dim // 2)  # noqa
            stacked = torch.cat([linear, -linear], dim=-1)  # (num_queries, num_batch, num_round, num_head, hash_dim)
            hashes = stacked.argmax(dim=-1, keepdim=False)  # (num_queries, num_batch, num_round, num_head)
            if seq_padding_mask is not None:
                mask_value = torch.tensor(self.num_hashes, dtype=torch.long).view(1, 1, 1, 1)
                hashes = torch.where(seq_padding_mask.transpose(0, 1).view(num_frames, num_batch, 1, 1), mask_value, hashes)  # noqa
            return hashes

        q_hashes = apply_hash(q, None)
        k_hashes = apply_hash(k, key_padding_mask)

        if override_hashes is not None:
            assert tuple(override_hashes.size()) == (num_queries, num_batch, self.num_rounds, self.num_heads)
            q_hashes, k_hashes = override_hashes, override_hashes

        q_hashes_sorted, q_sort_indices = torch.sort(q_hashes, dim=0, stable=True)
        k_hashes_sorted, k_sort_indices = torch.sort(k_hashes, dim=0, stable=True)

        assert tuple(q_sort_indices.size()) == (num_queries, num_batch, self.num_rounds, self.num_heads)
        assert tuple(k_sort_indices.size()) == (num_keys, num_batch, self.num_rounds, self.num_heads)

        index_range = torch.arange(0, num_queries).view(num_queries, 1, 1, 1).expand_as(q_sort_indices)
        q_inv_indices = q_sort_indices.new_empty(size=()).expand(num_queries, num_batch, self.num_rounds, self.num_heads)
        q_inv_indices = q_inv_indices.scatter(dim=0, index=q_sort_indices, src=index_range)
        assert tuple(q_inv_indices.size()) == tuple(q_sort_indices.size()) == (num_queries, num_batch, self.num_rounds, self.num_heads)  # noqa

        if not key_padding_mask:
            key_padding_mask = torch.zeros((num_batch, num_keys), dtype=torch.bool)
        assert tuple(key_padding_mask.size()) == (num_batch, num_keys)
        k_mask = torch.where(key_padding_mask.transpose(0, 1), float("-inf"), 0.0)  # (num_keys, num_batch)

        q = q.view(num_queries, num_batch, 1, self.num_heads, self.key_dim)
        k = k.view(num_keys, num_batch, 1, self.num_heads, self.key_dim)
        v = v.view(num_keys, num_batch, 1, self.num_heads, self.value_dim)
        k_mask = k_mask.view(num_keys, num_batch, 1, 1)
        q = q.expand(num_queries, num_batch, self.num_rounds, self.num_heads, self.key_dim)
        k = k.expand(num_keys, num_batch, self.num_rounds, self.num_heads, self.key_dim)
        v = v.expand(num_keys, num_batch, self.num_rounds, self.num_heads, self.value_dim)
        k_mask = k_mask.expand(num_keys, num_batch, self.num_rounds, self.num_heads)

        q_sorted = q.gather(dim=0, index=q_sort_indices.unsqueeze(-1).expand_as(q))
        k_sorted = k.gather(dim=0, index=k_sort_indices.unsqueeze(-1).expand_as(k))
        v_sorted = v.gather(dim=0, index=k_sort_indices.unsqueeze(-1).expand_as(v))
        k_mask_sorted = k_mask.gather(dim=0, index=k_sort_indices)

        num_query_chunks = self._ceildiv(num_queries, self.chunk_size)
        num_key_chunks = self._ceildiv(num_keys, self.chunk_size)

        q_sorted = F.pad(q_sorted, (0, 0) * 4 + (0, num_query_chunks * self.chunk_size - num_queries))
        q_sort_indices = F.pad(q_sort_indices, (0, 0) * 3 + (0, num_query_chunks * self.chunk_size - num_queries))
        q_hashes_sorted = F.pad(q_hashes_sorted, (0, 0) * 3 + (0, num_query_chunks * self.chunk_size - num_queries), value=-1)
        k_sorted = F.pad(k_sorted, (0, 0) * 4 + (0, num_key_chunks * self.chunk_size - num_keys))
        v_sorted = F.pad(v_sorted, (0, 0) * 4 + (0, num_key_chunks * self.chunk_size - num_keys))
        k_mask_sorted = F.pad(k_mask_sorted, (0, 0) * 3 + (0, num_key_chunks * self.chunk_size - num_keys), value=float("-inf"))  # noqa
        k_sort_indices = F.pad(k_sort_indices, (0, 0) * 3 + (0, num_key_chunks * self.chunk_size - num_keys))
        k_hashes_sorted = F.pad(k_hashes_sorted, (0, 0) * 3 + (0, num_key_chunks * self.chunk_size - num_keys), value=-1)

        # chunking
        q_sorted = q_sorted.view(num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.key_dim)  # noqa
        q_sort_indices = q_sort_indices.view(num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        q_hashes_sorted = q_hashes_sorted.view(num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_sorted = k_sorted.view(num_key_chunks, 1, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.key_dim)  # noqa
        v_sorted = v_sorted.view(num_key_chunks, 1, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa
        k_mask_sorted = k_mask_sorted.view(num_key_chunks, 1, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_sort_indices = k_sort_indices.view(num_key_chunks, 1, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_hashes_sorted = k_hashes_sorted.view(num_key_chunks, 1, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_sorted = k_sorted.expand(num_key_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.key_dim)  # noqa
        v_sorted = v_sorted.expand(num_key_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa
        k_mask_sorted = k_mask_sorted.expand(num_key_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_sort_indices = k_sort_indices.expand(num_key_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        k_hashes_sorted = k_hashes_sorted.expand(num_key_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa

        chunk_align = torch.arange(num_query_chunks, dtype=torch.int64).view(num_query_chunks, 1)  # (num_query_chunks, offset)  # noqa
        chunk_align = chunk_align + torch.arange(start=-1, end=num_chunk_offsets - 1, dtype=torch.int64).view(1, num_chunk_offsets)  # noqa
        chunk_align_mask = torch.where(torch.logical_or(chunk_align.lt(torch.tensor(0)), chunk_align.gt(num_query_chunks - 1)), float("-inf"), 0.0)  # (num_query_chunks, offset)  # noqa
        assert tuple(chunk_align_mask.size()) == (num_query_chunks, num_chunk_offsets)
        chunk_align_mask = chunk_align_mask.view(num_query_chunks, 1, 1, 1, 1, num_chunk_offsets, 1)

        chunk_align = chunk_align.clamp(0, num_query_chunks - 1)
        chunk_align = chunk_align.view(num_query_chunks, num_chunk_offsets, 1, 1, 1, 1)
        chunk_align = chunk_align.expand(num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        chunk_align_k = chunk_align.unsqueeze(-1).expand(num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.key_dim)  # noqa
        chunk_align_v = chunk_align.unsqueeze(-1).expand(num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa

        stacked_k_sorted = k_sorted.gather(dim=0, index=chunk_align_k)
        stacked_v_sorted = v_sorted.gather(dim=0, index=chunk_align_v)
        stacked_k_mask_sorted = k_mask_sorted.gather(dim=0, index=chunk_align)
        stacked_k_sort_indices = k_sort_indices.gather(dim=0, index=chunk_align)
        stacked_k_hashes_sorted = k_hashes_sorted.gather(dim=0, index=chunk_align)
        assert tuple(stacked_k_sorted.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.key_dim)  # noqa
        assert tuple(stacked_v_sorted.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa
        assert tuple(stacked_k_mask_sorted.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        assert tuple(stacked_k_sort_indices.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
        assert tuple(stacked_k_hashes_sorted.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa

        energy_sorted = torch.einsum("cibrnf,cojbrnf->cibrnoj", q_sorted, stacked_k_sorted)
        assert tuple(energy_sorted.size()) == (num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, num_chunk_offsets, self.chunk_size)  # noqa

        energy_sorted += chunk_align_mask
        energy_sorted += stacked_k_mask_sorted.permute(0, 3, 4, 5, 1, 2).unsqueeze(1)

        if causal:
            causal_mask = torch.where(stacked_k_sort_indices.permute(0, 3, 4, 5, 1, 2).unsqueeze(1).gt(q_sort_indices.unsqueeze(5).unsqueeze(-1)), float("-inf"), 0.0)  # noqa
            energy_sorted += causal_mask

        if self.mask_different_hashes:
            mask_val = float("-inf") if self.mask_current else -10.0**8
            hash_mask = torch.where(stacked_k_hashes_sorted.permute(0, 3, 4, 5, 1, 2).unsqueeze(1).ne(q_hashes_sorted.unsqueeze(5).unsqueeze(-1)), mask_val, 0.0)  # noqa
            energy_sorted += hash_mask

        if self.mask_current:
            current_mask = torch.where(stacked_k_sort_indices.permute(0, 3, 4, 5, 1, 2).unsqueeze(1).eq(q_sort_indices.unsqueeze(5).unsqueeze(-1)), -10.0**8, 0.0)  # noqa
            energy_sorted += current_mask

        if self.mask_different_rounds and self.num_rounds > 1:
            # In each hash round, each query attends to a given key at most one time.
            # We ensure this using the masking before (chunk_align_mask).
            # Now, here we want to count how many times query q attends to k in ANY round, and then
            # reduce the attention energies by energy(i,j) -= ln(|{i attends to j in round r' | round r'}|).
            # Naively, to compute |{i attends to j in round r' | round r'}|, for all i, j in round r
            # we would iterate over all rounds r', find the query chunk in which i appears, find the key j within
            # the query chunk of r', and then check if the energy there is over some threshold.

            # However, we do an approximation to improve memory and assume
            # |{i attends to j in round r' | round r'}| = |{h(i, r') == h(j, r') | round r'}|,
            # i.e. we assume that once the hashes match, that we can attend
            # In particular, we assume that the window size is always large enough.
            assert self.mask_different_hashes, "not implemented"
            assert not causal, "not implemented"
            # TODO: what about mask_current? what about chunk_align_mask?
            # TODO: can I instead of the hashes gather the energy that that key/query pair receives in the other hash round?

            def _gather_hashes_for_other_round(qk_sort_indices, qk_hashes):
                # q_sort_indices is (num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads)
                # q_hashes is (num_queries, num_batch, self.num_rounds, self.num_heads)
                # stacked_k_sort_indices is (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa
                # k_hashes is (num_keys, num_batch, self.num_rounds, self.num_heads)
                idx = qk_sort_indices
                idx = idx.unsqueeze(-1)
                idx = idx.expand(*qk_sort_indices.shape, self.num_rounds)  # (num_query_chunks, chunk_size, num_batch, num_rounds, num_heads, other_round)  # noqa
                hashes = qk_hashes
                hashes = hashes.transpose(-1, -2)  # (num_queries, num_batch, num_heads, num_rounds)
                hashes = hashes.unsqueeze(2)  # (num_queries, num_batch, 1=num_rounds, num_heads, other_round)
                for i in range(qk_sort_indices.ndim - qk_hashes.ndim):
                    hashes = hashes.unsqueeze(1)
                num_time = qk_hashes.size(0)
                hashes = hashes.expand(num_time, *idx.shape[1:])  # (num_queries, chunk_size, num_batch, num_rounds, num_heads, other_round)  # noqa
                other_round_hashes = hashes.gather(dim=0, index=idx)  # (num_query_chunks, chunk_size, num_batch, num_rounds, num_heads, other_round)  # noqa
                assert tuple(other_round_hashes.size()) == tuple(qk_sort_indices.size()) + (self.num_rounds,)
                return other_round_hashes

            other_round_q_hashes = _gather_hashes_for_other_round(q_sort_indices, q_hashes)
            assert tuple(other_round_q_hashes.size()) == (num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.num_rounds)
            other_round_k_hashes = _gather_hashes_for_other_round(k_sort_indices, k_hashes)
            assert tuple(other_round_k_hashes.size()) == (num_query_chunks, num_chunk_offsets, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.num_rounds)

            other_round_q_hashes = other_round_q_hashes.view(num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, 1, 1, self.num_rounds)
            other_round_k_hashes = other_round_k_hashes.permute(0, 3, 4, 5, 1, 2, 6).view(num_query_chunks, 1, num_batch, self.num_rounds, self.num_heads, num_chunk_offsets, self.chunk_size, self.num_rounds)

            hash_duplicate_counts = other_round_q_hashes.eq(other_round_k_hashes).to(torch.float).sum(dim=-1, keepdim=False)  # noqa
            assert tuple(hash_duplicate_counts.size()) == (num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, num_chunk_offsets, self.chunk_size)  # noqa

            round_mask = -torch.log(torch.clamp(hash_duplicate_counts, min=1))
            energy_sorted += round_mask

        energy_sorted_flat = energy_sorted.view(num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, num_chunk_offsets * self.chunk_size)  # noqa
        energy_lse_sorted = torch.logsumexp(energy_sorted_flat, dim=-1, keepdim=False)
        weights_sorted = torch.exp(energy_sorted - energy_lse_sorted.unsqueeze(-1).unsqueeze(-1))
        dropped_weights_sorted = self.dropout_module(weights_sorted)
        energy_lse_sorted = energy_lse_sorted.view(num_query_chunks * self.chunk_size, num_batch, self.num_rounds, self.num_heads)  # noqa

        round_out_sorted = torch.einsum("cibrnoj,cojbrnf->cibrnf", dropped_weights_sorted, stacked_v_sorted)
        assert tuple(round_out_sorted.size()) == (num_query_chunks, self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa
        round_out_sorted = round_out_sorted.reshape(num_query_chunks * self.chunk_size, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa

        q_inv_indices_v = q_inv_indices.unsqueeze(-1).expand(num_queries, num_batch, self.num_rounds, self.num_heads, self.value_dim)  # noqa
        round_out = round_out_sorted.gather(dim=0, index=q_inv_indices_v)
        assert tuple(round_out.size()) == (num_queries, num_batch, self.num_rounds, self.num_heads, self.value_dim)
        energy_lse = energy_lse_sorted.gather(dim=0, index=q_inv_indices)
        assert tuple(energy_lse.size()) == (num_queries, num_batch, self.num_rounds, self.num_heads)
        out = torch.sum(round_out * torch.softmax(energy_lse, dim=2).unsqueeze(-1), dim=2)
        assert tuple(out.size()) == (num_queries, num_batch, self.num_heads, self.value_dim)
        out = out.view(num_queries, num_batch, self.num_heads * self.value_dim)
        out = self.out_proj(out)

        assert not need_weights, "not implemented"
        out_weights = None

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
