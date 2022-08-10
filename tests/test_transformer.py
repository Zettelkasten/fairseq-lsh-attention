import argparse
import unittest
from typing import Any, Dict, Sequence

import torch
from fairseq.models import transformer
from fairseq.modules.multihead_lsh_attention import MultiheadLshAttention
from fairseq.modules.multihead_vanilla_attention import MultiheadVanillaAttention
from fairseq.sequence_generator import SequenceGenerator

from tests.test_roberta import FakeTask

import numpy as np


def mk_sample(num_batch: int = 2, max_length: int = 7) -> Dict[str, Any]:
    torch.manual_seed(42)
    src_lengths = torch.randint(low=3, high=max_length, size=(num_batch,))
    tokens = torch.stack([torch.arange(2, max_length + 2, dtype=torch.long)] * num_batch)
    tokens = torch.where(torch.arange(max_length).view(1, -1) < src_lengths.view(-1, 1), tokens, 1)

    sample = {
        "net_input": {
            "src_tokens": tokens,
            "prev_output_tokens": tokens,
            "src_lengths": src_lengths,
        },
        "target": tokens[:, 1:],
    }
    return sample


def mk_transformer(**extra_args: Any):
    overrides = {
        # Use characteristics dimensions
        "encoder_embed_dim": 12,
        "encoder_ffn_embed_dim": 14,
        "decoder_embed_dim": 12,
        "decoder_ffn_embed_dim": 14,
        # Disable dropout so we have comparable tests.
        "dropout": 0,
        "attention_dropout": 0,
        "activation_dropout": 0,
        "encoder_layerdrop": 0,
    }
    overrides.update(extra_args)
    # Overrides the defaults from the parser
    args = argparse.Namespace(**overrides)
    transformer.tiny_architecture(args)

    torch.manual_seed(0)
    task = FakeTask(args)
    return transformer.TransformerModel.build_model(args, task)


class TransformerTestCase(unittest.TestCase):
    def test_forward_backward(self):
        model = mk_transformer(encoder_embed_dim=12, decoder_embed_dim=12)
        sample = mk_sample()
        o, _ = model.forward(**sample["net_input"])
        loss = o.sum()
        loss.backward()

    def test_different_encoder_decoder_embed_dim(self):
        model = mk_transformer(encoder_embed_dim=12, decoder_embed_dim=16)
        sample = mk_sample()
        o, _ = model.forward(**sample["net_input"])
        loss = o.sum()
        loss.backward()


class TransformerLshTestCase(unittest.TestCase):
    def test_forward_backward_generate(self):
        num_heads = 5
        model_dim = num_heads * 3
        num_rounds = 1
        beam_size = 7
        num_hashes = 4
        chunk_size = 10
        model = mk_transformer(
            encoder_layers=1, decoder_layers=1,
            encoder_embed_dim=model_dim, decoder_embed_dim=model_dim,
            encoder_attention_heads=num_heads,
            decoder_attention_heads=num_heads,
            encoder_lsh_self_attn={"num_rounds": num_rounds, "num_hashes": num_hashes, "chunk_size": chunk_size},
            decoder_lsh_self_attn={"num_rounds": num_rounds, "num_hashes": num_hashes, "chunk_size": chunk_size},
            decoder_lsh_cross_attn={"num_rounds": num_rounds, "num_hashes": num_hashes, "chunk_size": chunk_size},
        )
        sample = mk_sample(num_batch=2)

        # test forward pass
        o, _ = model.forward(**sample["net_input"])
        print(f"Output is: {o}")

        # test backward pass
        loss = o.sum()
        loss.backward()

        # test beam search
        task = FakeTask(args={})
        generator = SequenceGenerator([model], task.dictionary, beam_size=beam_size)
        hypos = generator.forward(sample)
        print(f"Hypotheses are: {hypos}")

    def test_lsh_attention_equal_to_full_attention(self):
        # Test that full attention and lsh attention are equal when the chunk size is large enough
        # and we allow attention to keys of different hash classes.

        def run_full_and_lsh(*,
                             num_heads = 1, kv_dim = 1, num_rounds = 1, num_hashes = 16, chunk_size = 5,
                             self_attention=True, causal=False,
                             num_batch=1, num_time=10, dynamic_time = False):
            embed_dim = num_heads * kv_dim
            assert num_time <= 3 * chunk_size, "chunk size must be large enough for this test to work"
            num_queries = num_time
            num_keys = num_time if self_attention else num_time - 1  # just to have some variation

            torch.manual_seed(42)
            full_att = MultiheadVanillaAttention(
                embed_dim=embed_dim, num_heads=num_heads, self_attention=self_attention,
                encoder_decoder_attention=not self_attention
            )
            lsh_att = MultiheadLshAttention(
                embed_dim=embed_dim, num_heads=num_heads, self_attention=self_attention,
                encoder_decoder_attention=not self_attention,
                num_rounds=num_rounds, num_hashes=num_hashes, chunk_size=chunk_size,
                share_kq=False, mask_different_hashes=False
            )
            lsh_att.q_proj = full_att.q_proj
            lsh_att.k_proj = full_att.k_proj
            lsh_att.v_proj = full_att.v_proj
            lsh_att.out_proj = full_att.out_proj

            query = 2 * torch.rand((num_queries, num_batch, embed_dim)) - 1
            if self_attention:
                key = query
            else:
                key = 2 * torch.rand((num_keys, num_batch, embed_dim)) - 1
            attn_mask = True if causal else None
            if dynamic_time:
                query_seq_lengths = torch.randint(num_queries // 2, num_queries - 1, size=(num_batch,))
                if self_attention:
                    key_seq_lengths = query_seq_lengths
                else:
                    key_seq_lengths = torch.randint(num_keys // 2, num_keys - 1, size=(num_batch,))
                query_mask = torch.arange(num_queries).view(1, -1).gt(query_seq_lengths.view(-1, 1)).to(torch.bool)
                key_mask = torch.arange(num_keys).view(1, -1).gt(key_seq_lengths.view(-1, 1)).to(torch.bool)
            else:
                query_mask, key_mask = None, None
            full_out, full_vars = full_att(
                query=query, key=key, value=key, need_weights=False, attn_mask=attn_mask,
                query_padding_mask=query_mask, key_padding_mask=key_mask
            )
            lsh_out, _ = lsh_att(
                query=query, key=key, value=key, need_weights=False, attn_mask=attn_mask,
                query_padding_mask=query_mask, key_padding_mask=key_mask,
                need_head_weights=full_vars  # for debugging, s.t. we can compare intermediate values easily
            )

            return full_out, lsh_out

        cases = {
            "single_head_single_round_no_batch_full_time": {
                "num_heads": 1, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 1, "num_time": 10, "dynamic_time": False
            },
            "single_head_single_round_no_batch": {
                "num_heads": 1, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 1, "num_time": 7, "dynamic_time": False
            },
            "single_round_no_batch": {
                "num_heads": 4, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 1, "num_time": 7, "dynamic_time": False
            },
            "single_head_single_round_small_dim": {
                "num_heads": 1, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 3, "num_time": 7, "dynamic_time": False
            },
            "single_round_small_dim": {
                "num_heads": 4, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 3, "num_time": 7, "dynamic_time": False
            },
            "single_round": {
                "num_heads": 8, "kv_dim": 64, "num_rounds": 1, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 3, "num_time": 10, "dynamic_time": False
            },
            "multi_round": {
                "num_heads": 8, "kv_dim": 64, "num_rounds": 6, "num_hashes": 16, "chunk_size": 5,
                "self_attention": True,
                "num_batch": 3, "num_time": 10, "dynamic_time": False
            },
            "single_round_causal": {
                "num_heads": 8, "kv_dim": 64, "num_rounds": 1, "num_hashes": 16, "chunk_size": 10,
                "self_attention": True, "causal": True,
                "num_batch": 3, "num_time": 10, "dynamic_time": False
            },
            "multi_round_causal": {
                "num_heads": 8, "kv_dim": 64, "num_rounds": 6, "num_hashes": 16, "chunk_size": 10,
                "self_attention": True, "causal": True,
                "num_batch": 3, "num_time": 10, "dynamic_time": False
            },
            "simple_dynamic_time": {
                "num_heads": 1, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 10,
                "self_attention": True,
                "num_batch": 1, "num_time": 10, "dynamic_time": True
            },
            "longer_dynamic_time": {
                "num_heads": 8, "kv_dim": 64, "num_rounds": 1, "num_hashes": 16, "chunk_size": 10,
                "self_attention": True,
                "num_batch": 5, "num_time": 10, "dynamic_time": True
            },
            "simple_cross_att": {
                "num_heads": 1, "kv_dim": 1, "num_rounds": 1, "num_hashes": 16, "chunk_size": 10,
                "self_attention": False,
                "num_batch": 1, "num_time": 10, "dynamic_time": True
            },
        }

        for case_name, case_params in cases.items():
            with self.subTest(msg=case_name, **case_params):
                print(f"=== Executing {case_name}")
                print(f"Params: {case_params}")
                full_out, lsh_out = run_full_and_lsh(**case_params)
                print(f"Lsh att output: {lsh_out}")
                print(f"Full att output: {full_out}")
                assert lsh_out.shape == full_out.shape
                torch.testing.assert_allclose(lsh_out, full_out)

    def test_lsh_attention_hashing(self):
        # For input position i, set key = value = i-th unit vector.
        # Distance between all query-key pairs is equal this way.
        # Set chunk size large enough s.t. only different hash classes will cause pruning.

        def do_test(*, hash_sequence, chunk_size, causal: bool, num_hashes=42):
            hash_sequence = torch.tensor(hash_sequence)
            assert len(hash_sequence.shape) in [1, 2], "[time] or [round,time]"
            if len(hash_sequence.shape) == 1:
                hash_sequence = hash_sequence.unsqueeze(0)
            num_rounds, num_time = hash_sequence.shape
            feat_dim = num_time

            qkv = torch.zeros(num_time, 1, feat_dim)
            for t in range(num_time):
                qkv[t, :, t] = 1.0

            lsh_att = MultiheadLshAttention(
                embed_dim=feat_dim, num_heads=1, self_attention=True,
                num_rounds=num_rounds, num_hashes=num_hashes, chunk_size=chunk_size, mask_current=True,
                share_kq=True
            )

            lsh_att.q_proj = torch.nn.Identity()
            lsh_att.k_proj = torch.nn.Identity()
            lsh_att.v_proj = torch.nn.Identity()
            lsh_att.out_proj = torch.nn.Identity()

            attn_mask = True if causal else None
            lsh_out, _ = lsh_att(
                query=qkv, key=qkv, value=qkv, need_weights=False, attn_mask=attn_mask,
                override_hashes=hash_sequence.transpose(0, 1).view(num_time, 1, num_rounds, 1))

            print("Hash sequence:")
            print(hash_sequence.squeeze())

            print("Lsh output:")
            print(lsh_out.squeeze())

            assert tuple(lsh_out.shape) == (num_time, 1, feat_dim)
            for query_t in range(num_time):
                attended_keys = [
                    key_t
                    for key_t in range(num_time)
                    if any(hash_sequence[r, key_t] == hash_sequence[r, query_t] for r in range(num_rounds))
                    and key_t != query_t
                    and (not causal or key_t <= query_t)
                ]
                if len(attended_keys) == 0:
                    attended_keys = [query_t]
                target = torch.zeros(feat_dim)
                for key_t in attended_keys:
                    target[key_t] = 1.0 / len(attended_keys)

                if not torch.allclose(lsh_out[query_t, 0], target):
                    print(f"time = {query_t} mismatches!")
                    print(f"got output: {lsh_out[query_t, 0]}")
                    print(f"but expected: {target}")
                    print(f"complete hash sequence was: {hash_sequence}")
                torch.testing.assert_allclose(lsh_out[query_t, 0], target)

        torch.manual_seed(42)
        rand_1_30 = torch.randint(0, 26, size=(30,))
        rand_2_30 = torch.randint(0, 36, size=(2, 30))
        rand_10_30 = torch.randint(0, 36, size=(10, 30))

        cases = {
            "single_chunk": {
                "hash_sequence": [1,1,1,2,2,2,3,3,3], "chunk_size": 10, "causal": False
            },
            "single_chunk_causal": {
                "hash_sequence": [1,1,1,2,2,2,3,3,3], "chunk_size": 10, "causal": True
            },
            "single_chunk2": {
                "hash_sequence": [1,2,3,4,5,6,6,6,7,8,8,8,9], "chunk_size": 15, "causal": False
            },
            "single_chunk2_causal": {
                "hash_sequence": [1,2,3,4,5,6,6,6,7,8,8,8,9], "chunk_size": 15, "causal": True
            },
            "two_chunks": {
                "hash_sequence": [2,2,1,1,1], "chunk_size": 3, "causal": False
            },
            "two_chunks_causal": {
                "hash_sequence": [2,2,1,1,1], "chunk_size": 3, "causal": True
            },
            "chunk_range": {
                "hash_sequence": np.arange(20), "chunk_size": 3, "causal": False
            },
            "chunk_range_inverse": {
                "hash_sequence": np.flip(np.arange(20)).copy(), "chunk_size": 3, "causal": False
            },
            "chunk_staggered": {
                "hash_sequence": [9,9,1,1,8,8,2,2,7,7,3,3,6,6,4,4,5,5], "chunk_size": 3, "causal": False
            },
            "chunk_staggered_causal": {
                "hash_sequence": [9,9,1,1,8,8,2,2,7,7,3,3,6,6,4,4,5,5], "chunk_size": 3, "causal": True
            },
            # technically, the chunk size is too small. but it is very unlikely that more than 3 keys have the same hash
            "big": {
                "hash_sequence": rand_1_30, "chunk_size": 3, "causal": False
            },
            "big_causal": {
                "hash_sequence": rand_1_30, "chunk_size": 3, "causal": True
            },
            # hash rounds do not help here, as the hash classes for all rounds are equal
            "multi_round_single_chunk_equal": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [1,1,1,2,2,2,3,3,3]], "chunk_size": 10, "causal": False
            },
            "multi_round_single_chunk_equal_causal": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [1,1,1,2,2,2,3,3,3]], "chunk_size": 10, "causal": True
            },
            "multi_round_single_chunk_equal2": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [3,3,3,2,2,2,1,1,1]], "chunk_size": 10, "causal": False
            },
            "multi_round_single_chunk_equal2_causal": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [3,3,3,2,2,2,1,1,1]], "chunk_size": 10, "causal": True
            },
            "multi_round_single_chunk": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [5,5,5,5,5,5,5,5,5]], "chunk_size": 10, "causal": False
            },
            "multi_round_single_chunk_causal": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [5,5,5,5,5,5,5,5,5]], "chunk_size": 10, "causal": True
            },
            # hash rounds should now increase the effective window, but keys of different hash rounds are disjoint.
            "multi_round_disjoint": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [1,2,3,4,1,2,3,4,5]], "chunk_size": 10, "causal": False
            },
            "multi_round_disjoint_causal": {
                "hash_sequence": [[1,1,1,2,2,2,3,3,3], [1,2,3,4,1,2,3,4,5]], "chunk_size": 10, "causal": True
            },
            # hash rounds increase the effective window, but also select keys twice some times.
            "multi_round_duplicates": {
                "hash_sequence": [[1,2,2,2], [5,5,5,6]], "chunk_size": 4, "causal": False
            },
            "multi_round_duplicates_causal": {
                "hash_sequence": [[1,2,2,2], [5,5,5,6]], "chunk_size": 4, "causal": True
            },
            "multi_round_big": {
                "hash_sequence": rand_2_30, "chunk_size": 4, "causal": False
            },
            "multi_round_big_causal": {
                "hash_sequence": rand_2_30, "chunk_size": 4, "causal": True
            },
            "multi_round_big_many_rounds": {
                "hash_sequence": rand_10_30, "chunk_size": 4, "causal": False
            },
            "multi_round_big_many_rounds_causal": {
                "hash_sequence": rand_10_30, "chunk_size": 4, "causal": True
            }
        }

        for case_name, case_params in cases.items():
            with self.subTest(msg=case_name, **case_params):
                print(f"=== Executing {case_name}")
                print(f"Params: {case_params}")
                do_test(**case_params)
