import pytest
import torch
import math
from unittest.mock import patch, MagicMock

from mage_flow.pipeline import _velocity, _build_pack_ctx, build_scheduler
from mage_flow.models.modules._attn_backend import _resolve_fa4, _resolve_sdpa

def test_cfg_global_norm_normalization():
    ctx = {
        "cfg": 5.0,
        "renorm": True,
        "batch_cfg": False,
        "has_neg": True,
        "img_ids": None, "img_cu": None, "img_max": None, "img_shapes": None,
        "txt": None, "txt_ids": None, "txt_cu": None, "txt_mask": None, "txt_max": None, "vec": None,
        "neg_txt": None, "neg_ids": None, "neg_cu": None, "neg_mask": None, "neg_max": None, "neg_vec": None,
        "na": 1
    }
    
    torch.manual_seed(42)
    cond = torch.randn(1, 1024, 128)
    unc = torch.randn(1, 1024, 128)
    
    def mock_transformer(**kwargs):
        if kwargs.get("txt") is None:
            return unc
        return cond
        
    out = _velocity(mock_transformer, torch.zeros(1, 1024, 128), ctx, 0.5)
    
    cond_norm = torch.norm(cond.float(), dim=-1, keepdim=True).mean()
    out_norm = torch.norm(out.float(), dim=-1, keepdim=True).mean()
    
    assert torch.isclose(cond_norm, out_norm, rtol=1e-3)
    
    comb = unc + 5.0 * (cond - unc)
    out_dir = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-6)
    comb_dir = comb / (torch.norm(comb, dim=-1, keepdim=True) + 1e-6)
    
    assert torch.allclose(out_dir, comb_dir, atol=1e-4)

def test_fa4_hardware_guard():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
            with patch("mage_flow.models.modules._attn_backend._resolve_fa2") as mock_fa2:
                mock_fa2.return_value = "FA2_FALLBACK"
                fn = _resolve_fa4()
                assert fn == "FA2_FALLBACK"
                mock_fa2.assert_called_once()
                
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            # Mocking the lazy import to prevent ImportError
            import sys
            cute_mock = MagicMock()
            cute_mock.flash_attn_varlen_func = "FA4_FUNC"
            sys.modules["flash_attn.cute"] = cute_mock
            
            fn = _resolve_fa4()
            assert callable(fn)

def test_vram_aware_batch_cfg():
    class MockTransformer:
        num_heads = 28
        head_dim = 128
        
    transformer = MockTransformer()
    img_lens = [4096, 4096]
    txt_cu = torch.tensor([0, 10, 20])
    neg_cu = torch.tensor([0, 10, 20])
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    with patch("torch.cuda.mem_get_info", return_value=(20 * 1024**3, 24 * 1024**3)):
        # Force cuda device to trigger memory check even if running on cpu machine
        # by passing a mock device that looks like cuda
        mock_dev = MagicMock()
        mock_dev.type = "cuda"
        
        ctx = _build_pack_ctx(
            transformer, None, None, [None], img_lens, 
            torch.zeros(1, 20, 10), txt_cu, None, torch.zeros(2, 10),
            torch.zeros(1, 20, 10), neg_cu, None, torch.zeros(2, 10),
            5.0, False, True, mock_dev
        )
        assert ctx["batch_cfg"] is False

def test_turbo_scheduler_sigmas():
    sched = build_scheduler(num_steps=4, shift=1.0)
    sigmas = sched.timesteps
    
    # Expected cosine sigmas:
    # steps = [0, pi/6, pi/3, pi/2]
    # sin(steps) = [0, 0.5, 0.866, 1.0]
    # 1 - sin(steps[:-1]) = [1.0, 0.5, 0.134, 0.0]  (terminal 0 is appended by set_timesteps maybe)
    assert len(sigmas) == 5
    assert torch.isclose(sigmas[0], torch.tensor(1.0))
    assert torch.isclose(sigmas[1], torch.tensor(0.5))
    assert sigmas[-1] == 0.0
