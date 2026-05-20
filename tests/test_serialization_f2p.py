import torch
import pytest
import bitsandbytes as bnb

def test_linear4bit_state_dict_roundtrip():
    src = bnb.nn.Linear4bit(
        64, 64, bias=False, quant_type='nf4', 
        compute_dtype=torch.float32, compress_statistics=True
    )
    sd = src.state_dict()
    dst = bnb.nn.Linear4bit(
        64, 64, bias=False, quant_type='nf4', 
        compute_dtype=torch.float32, compress_statistics=True
    )
    try:
        dst.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        pytest.fail(f"load_state_dict failed under strict verification: {e}")
