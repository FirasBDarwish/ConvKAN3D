import pytest
from unittest.mock import patch
from ConvKAN3D import ConvKAN3D
import torch

conv = ConvKAN3D(in_channels=3, out_channels=3, kernel_size=3)
x = torch.randn((64,3,9,9,9))
y = conv(x)
print(y.shape)
