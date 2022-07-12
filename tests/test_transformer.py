import argparse
import unittest
from typing import Any, Dict, Sequence

import torch
from fairseq.models import transformer
from fairseq.modules.multihead_lsh_attention import MultiheadLshAttention
from fairseq.modules.multihead_vanilla_attention import MultiheadVanillaAttention
from fairseq.sequence_generator import SequenceGenerator

from tests.test_roberta import FakeTask


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

        def run_full_and_lsh(*,
                             num_heads = 1, kv_dim = 1, num_rounds = 1, num_hashes = 16, chunk_size = 5,
                             self_attention=True,
                             num_batch=1, num_time=10, dynamic_time = False):
            embed_dim = num_heads * kv_dim

            torch.manual_seed(42)
            full_att = MultiheadVanillaAttention(
                embed_dim=embed_dim, num_heads=num_heads, self_attention=self_attention
            )
            lsh_att = MultiheadLshAttention(
                embed_dim=embed_dim, num_heads=num_heads, self_attention=self_attention,
                num_rounds=num_rounds, num_hashes=num_hashes, chunk_size=chunk_size,
                share_kq=False
            )
            lsh_att.q_proj = full_att.q_proj
            lsh_att.k_proj = full_att.k_proj
            lsh_att.v_proj = full_att.v_proj
            lsh_att.out_proj = full_att.out_proj

            query = 2 * torch.rand((num_time, num_batch, embed_dim)) - 1
            full_out, full_vars = full_att(query=query, key=query, value=query, need_weights=False)
            lsh_out, _ = lsh_att(query=query, key=query, value=query, need_weights=False, attn_mask=full_vars)

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
        }

        for case_name, case_params in cases.items():
            with self.subTest(msg=case_name, **case_params):
                print(f"Params: {case_params}")
                full_out, lsh_out = run_full_and_lsh(**case_params)
                print(f"Lsh att output: {lsh_out}")
                print(f"Full att output: {full_out}")
                assert lsh_out.shape == full_out.shape
                torch.testing.assert_allclose(lsh_out, full_out)
