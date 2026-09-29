"""Small GPU correctness check, never a performance measurement."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness test")
def test_nano_gqa_joint_attention_and_incoming_score():
    from piper_cosmos.cosmos3.nano_cache_vendor.toca_joint_attention import joint_attention_score
    torch.manual_seed(11)
    q = torch.randn(37, 32, 128, device="cuda", dtype=torch.bfloat16)
    kg = torch.randn(37, 8, 128, device="cuda", dtype=torch.bfloat16)
    vg = torch.randn_like(kg)
    ku = torch.randn(13, 8, 128, device="cuda", dtype=torch.bfloat16)
    vu = torch.randn_like(ku)
    future = torch.arange(5, 29, device="cuda", dtype=torch.long)
    out, score = joint_attention_score(q, ku, kg, vu, vg, future, 128**-.5)
    k = torch.cat((ku, kg)).repeat_interleave(4, 1).transpose(0, 1).float()
    v = torch.cat((vu, vg)).repeat_interleave(4, 1).transpose(0, 1).float()
    prob = (q.transpose(0, 1).float() @ k.transpose(-1, -2) * 128**-.5).softmax(-1)
    expected = (prob @ v).transpose(0, 1)
    expected_score = prob[:, :, future+13].mean((0, 1))
    torch.testing.assert_close(out.float(), expected, atol=.01, rtol=.03)
    torch.testing.assert_close(score, expected_score, atol=1e-5, rtol=.002)
    assert torch.isfinite(out).all() and torch.isfinite(score).all()
