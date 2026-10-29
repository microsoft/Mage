import pytest
import torch
import torch.nn as nn

from mage_flow.models.utils import optionally_expand_state_dict, validate_state_dict_keys
from mage_flow.models.modules.mage_text import _full_output_mode


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(16, 64)
        self.txt_in = nn.Linear(32, 64)
        self.proj_out = nn.Linear(64, 16)
        self.non_critical = nn.Linear(10, 10)


def test_validate_state_dict_keys_missing_critical():
    model = MockModel()
    state_dict = model.state_dict()
    # Remove a critical layer
    del state_dict["img_in.weight"]
    
    with pytest.raises(KeyError, match="Critical layer 'img_in.weight' is missing"):
        validate_state_dict_keys(model, state_dict, strict_critical=True)


def test_validate_state_dict_keys_success():
    model = MockModel()
    state_dict = model.state_dict()
    # Should not raise
    missing, unexpected = validate_state_dict_keys(model, state_dict, strict_critical=True)
    assert len(missing) == 0


def test_optionally_expand_state_dict_critical_projection():
    model = MockModel()
    state_dict = model.state_dict()
    
    # Create a shape mismatch on a critical layer
    state_dict["img_in.weight"] = torch.randn(64, 8) # Expected is 64, 16
    
    with pytest.raises(ValueError, match="Zero-padding critical projection weights produces dead feature channels"):
        optionally_expand_state_dict(model, state_dict)


def test_optionally_expand_state_dict_non_critical():
    model = MockModel()
    state_dict = model.state_dict()
    
    # Create a shape mismatch on a non-critical layer
    state_dict["non_critical.weight"] = torch.randn(5, 5) # Expected 10, 10
    
    # Should not raise and should expand
    expanded_dict = optionally_expand_state_dict(model, state_dict)
    assert expanded_dict["non_critical.weight"].shape == (10, 10)


def test_full_output_mode_exception_reraise():
    class MockHF:
        def __init__(self):
            self._output_mode = "embedding"
            self._skip_lm_head = True
            
        def set_output_mode(self, mode):
            # simulate an error during restore
            if mode == "embedding":
                raise RuntimeError("Simulated failure in set_output_mode")
            self._output_mode = mode

    hf_model = MockHF()
    
    with pytest.raises(RuntimeError, match="Simulated failure in set_output_mode"):
        with _full_output_mode(hf_model):
            pass # Exception should be raised when exiting the context manager
