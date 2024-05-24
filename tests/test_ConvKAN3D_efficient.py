""" Testing the ConvKAN3D (EfficientKAN) Module """

import pytest
from unittest.mock import patch
from src.ConvKAN3D.ConvKAN3D import ConvKAN3D
import torch

class Tests:
    """ Class defines tests """

    def test_assertions(self):
        conv = ConvKAN3D(in_channels=3, out_channels=3, kernel_size=3)
        assert str(type(conv)) == "<class 'ConvKAN3D.ConvKAN3D.ConvKAN3D'>"

    def test_pass(self):
        conv = ConvKAN3D(in_channels=3, out_channels=3, kernel_size=3)
        x = torch.randn((64,3,9,9,9))
        y = conv(x)
        assert str(type(y)) == "<class 'torch.Tensor'>"
        assert str(y.shape) == "torch.Size([64, 3, 7, 7, 7])"