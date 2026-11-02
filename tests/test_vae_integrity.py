import pytest
import torch
import torch.nn as nn

from mage_flow.models.modules.mage_vae import _ConstAdaLN
from mage_flow.models.utils import pad_to_patch_multiple


def test_pad_to_patch_multiple():
    # Test tensor of shape [1, 3, 50, 50] padded to multiple of 16
    x = torch.randn(1, 3, 50, 50)
    x_pad, (pad_h, pad_w) = pad_to_patch_multiple(x, patch_size=16)
    
    assert x_pad.shape[-2] == 64
    assert x_pad.shape[-1] == 64
    assert pad_h == 14
    assert pad_w == 14
    
    # Test already multiple
    y = torch.randn(1, 3, 32, 32)
    y_pad, (pad_h, pad_w) = pad_to_patch_multiple(y, patch_size=16)
    assert y_pad.shape[-2] == 32
    assert y_pad.shape[-1] == 32
    assert pad_h == 0
    assert pad_w == 0


def test_deterministic_encoding():
    class MockVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.sample_posterior = True
            
            class MockEncoder:
                patch_size = 16
            self.dconv_encoder = MockEncoder()
            
        def _encode_moments(self, x):
            B, C, H, W = x.shape
            mean = torch.zeros(B, 128, H // 16, W // 16)
            logvar = torch.zeros(B, 128, H // 16, W // 16)
            return mean, logvar
            
    vae = MockVAE()
    from mage_flow.models.modules.mage_vae import MageVAE as ActualMageVAE
    vae.encode = ActualMageVAE.encode.__get__(vae)
    
    x = torch.randn(1, 3, 64, 64)
    
    gen1 = torch.Generator().manual_seed(42)
    out1 = vae.encode(x, generator=gen1)
    
    gen2 = torch.Generator().manual_seed(42)
    out2 = vae.encode(x, generator=gen2)
    
    gen3 = torch.Generator().manual_seed(43)
    out3 = vae.encode(x, generator=gen3)
    
    assert torch.allclose(out1, out2)
    assert not torch.allclose(out1, out3)


def test_fp32_precision_guard():
    class MockVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.sample_posterior = True
            class MockEncoder:
                patch_size = 16
            self.dconv_encoder = MockEncoder()
            
        def _encode_moments(self, x):
            mean = torch.zeros(1, 128, 4, 4, dtype=torch.bfloat16)
            logvar = torch.full((1, 128, 4, 4), -20.0, dtype=torch.bfloat16)
            return mean, logvar
            
    vae = MockVAE()
    from mage_flow.models.modules.mage_vae import MageVAE as ActualMageVAE
    vae.encode = ActualMageVAE.encode.__get__(vae)
    
    x = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
    
    gen = torch.Generator().manual_seed(0)
    out = vae.encode(x, generator=gen)
    
    assert not torch.isnan(out).any()
    assert out.dtype == torch.bfloat16


def test_dynamic_adaln_cache():
    original_mlp = nn.Linear(4, 4)
    nn.init.constant_(original_mlp.weight, 1.0)
    nn.init.constant_(original_mlp.bias, 1.0)
    modulation = torch.zeros(1, 4)
    
    const_adaln = _ConstAdaLN(modulation, original_mlp)
    c_t0 = torch.zeros(1, 4)
    
    # Enabled and no bypass -> returns modulation
    const_adaln.enabled = True
    const_adaln.bypass = False
    assert torch.allclose(const_adaln(c_t0), modulation)
    
    # Bypass -> returns original_mlp output
    const_adaln.bypass = True
    assert not torch.allclose(const_adaln(c_t0), modulation)
