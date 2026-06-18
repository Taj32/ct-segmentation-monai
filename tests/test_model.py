import torch
import numpy as np
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def test_model_forward_pass():
    """Test 3D U-Net forward pass with dummy input"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 3, 96, 96, 96)

def test_output_channels():
    """Test model has correct number of output classes"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[1] == 3  # background, liver, tumor

def test_softmax_output():
    """Test softmax probabilities sum to 1"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        output = model(dummy_input)
    probs = torch.softmax(output, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)