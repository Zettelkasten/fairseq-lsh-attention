import argparse
import unittest
from typing import Any, Dict, Sequence

import torch
from fairseq.models import transformer
from fairseq.sequence_generator import SequenceGenerator

from tests.test_roberta import FakeTask


def mk_sample(tok: Sequence[int] = None, batch_size: int = 2) -> Dict[str, Any]:
    if not tok:
        tok = [10, 11, 12, 13, 14, 15, 2]

    batch = torch.stack([torch.tensor(tok, dtype=torch.long)] * batch_size)
    sample = {
        "net_input": {
            "src_tokens": batch,
            "prev_output_tokens": batch,
            "src_lengths": torch.tensor(
                [len(tok)] * batch_size, dtype=torch.long, device=batch.device
            ),
        },
        "target": batch[:, 1:],
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
        sample = mk_sample(batch_size=2)

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
